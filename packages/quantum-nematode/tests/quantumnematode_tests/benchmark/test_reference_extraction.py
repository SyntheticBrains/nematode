"""The reference extraction: its readers, the store's encodings and the cohort split's rule.

Covers the benchmark-references spec's scenarios "The unpickler refuses other classes", "An arm
without a lawn boundary", "The split re-draws identically", "An arm with one date has no held-out
cohort", "An arm whose dates cannot meet the share", "No draw passes", "Calhoun's scale is
unstated" and "López-Cruz lacks a D37 mutant".
"""

from __future__ import annotations

import os
import pickle
import struct
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO / "scripts" / "analysis"))

import _scheer_pickle  # noqa: E402  # pyright: ignore[reportMissingImports]
import extract_benchmark_references as ex  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.benchmark import references as refs  # noqa: E402


class _Malicious:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, ("echo pwned",))


def test_the_unpickler_refuses_other_classes(tmp_path: Path) -> None:
    """A pickle that would call ``os.system`` is refused before anything runs."""
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(_Malicious()))
    with path.open("rb") as f, pytest.raises(pickle.UnpicklingError, match="refused"):
        _scheer_pickle.SafeUnpickler(f).load()


def test_pandas_1_numeric_indexes_read_as_index() -> None:
    """A pandas 1.x ``Int64Index`` resolves to ``pandas.Index``; other classes stay refused."""
    import io

    import pandas as pd

    unpickler = _scheer_pickle.SafeUnpickler(io.BytesIO(b""))
    assert unpickler.find_class("pandas.core.indexes.numeric", "Int64Index") is pd.Index
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        unpickler.find_class("os", "system")


def test_positions_round_trip_through_the_store() -> None:
    """Micrometre integers return as mm within half a micrometre, with gaps kept as NaN."""
    mm = np.array([0.0, 1.23456, -32.7, np.nan, 4.0004])
    back = refs.dequantise(ex.quantise_um(mm, np.int16), 1000.0)
    assert np.isnan(back[3])
    assert np.nanmax(np.abs(back - mm)) <= 0.0005


def test_wide_positions_choose_int32() -> None:
    """A position beyond int16's micrometre range moves the store to int32 rather than clipping."""
    assert ex.position_dtype([np.array([1.0, -32.0])]) == np.int16
    assert ex.position_dtype([np.array([1.0, 47.5])]) == np.int32


def test_flags_pack_and_unpack() -> None:
    """Each flag comes back from its own bit."""
    rng = np.random.default_rng(0)
    flags = [rng.random((3, 50)) < 0.3 for _ in range(6)]
    packed = ex.pack_flags(flags)
    for bit, flag in enumerate(flags):
        assert np.array_equal(refs.unpack_flag(packed, bit), flag)


def test_boundary_radii_of_a_circle() -> None:
    """A circle's polygon gives its radius at every angle."""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    radii = ex.boundary_radii(5 + 2 * np.cos(theta), -1 + 2 * np.sin(theta), 5.0, -1.0)
    assert radii.shape == (ex.BOUNDARY_ANGLES,)
    assert np.allclose(radii, 2.0, atol=1e-3)


def _arm(n: int, *, boundary: bool) -> dict[str, object]:
    import datetime as dt

    import pandas as pd

    rng = np.random.default_rng(1)
    frames = 30 * 4
    x = 100 + np.cumsum(rng.normal(0, 0.5, (n, frames)), axis=1)
    y = 200 + np.cumsum(rng.normal(0, 0.5, (n, frames)), axis=1)
    data: dict[str, object] = {
        "Midbody_cent_x": x,
        "Midbody_cent_y": y,
        "pixpermm": np.full((n, 1), 40.0),
        "videoname": pd.DataFrame([f"v{i}" for i in range(n)]),
        "matfilename": pd.DataFrame([f"m{i}" for i in range(n)]),
        "dates": pd.Series([dt.date(2021, 3, 1 + i % 2) for i in range(n)]),
        "MovingReverse": rng.random((n, frames)) < 0.1,
        "Omega": rng.random((n, frames)) < 0.01,
        "Pause": rng.random((n, frames)) < 0.1,
        "missingdata": np.zeros((n, frames), dtype=bool),
        "InLawnRunMask": np.ones((n, 4), dtype=bool),
        "OutLawnRunMask": np.zeros((n, 4), dtype=bool),
        "RD_states_Matrix_exog": np.ma.MaskedArray(
            np.ones((n, 4)),
            mask=np.zeros((n, 4), dtype=bool),
        ),
    }
    if boundary:
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        data["Center_Point_x"] = np.full((n, 1), 100.0)
        data["Center_Point_y"] = np.full((n, 1), 200.0)
        data["Lawn_Boundary_Pts_x"] = np.tile(100 + 60 * np.cos(theta), (n, 1))
        data["Lawn_Boundary_Pts_y"] = np.tile(200 + 60 * np.sin(theta), (n, 1))
        data["HeadPokesAll"] = np.zeros((n, frames), dtype=bool)
        data["MidbodyInLawn"] = np.ones((n, frames), dtype=bool)
    return data


def test_an_arm_with_a_boundary_is_centred_on_its_lawn() -> None:
    """Positions are from the lawn centre in mm, and the boundary radius is the lawn's."""
    arrays, notes = ex.scheer_arm(_arm(3, boundary=True), "PD1074_od2_Data")
    data = _arm(3, boundary=True)
    expected = (np.asarray(data["Midbody_cent_x"])[:, :: ex.POSITION_STRIDE] - 100.0) / 40.0
    assert np.allclose(arrays["x_mm"], expected)
    assert np.allclose(arrays["boundary_mm"], 1.5, atol=1e-3)
    assert notes["has_boundary"]
    assert notes["absent_flags"] == []
    assert np.all(arrays["origin"] == refs.ORIGIN_LAWN_CENTRE)


def test_an_arm_without_a_lawn_boundary() -> None:
    """Without a boundary, positions start at the origin and the radii are missing."""
    arrays, notes = ex.scheer_arm(_arm(3, boundary=False), "PD1074_od2_uniformlawn_Data")
    assert np.allclose(arrays["x_mm"][:, 0], 0.0)
    assert np.allclose(arrays["y_mm"][:, 0], 0.0)
    assert np.all(np.isnan(arrays["boundary_mm"]))
    assert np.all(arrays["origin"] == refs.ORIGIN_FIRST_POSITION)
    assert not notes["has_boundary"]
    assert set(notes["absent_flags"]) == {"HeadPokesAll", "MidbodyInLawn"}


def test_animals_match_by_video_and_file_in_order() -> None:
    """Repeated (video, file) pairs match in order of appearance; missing ones do not match."""
    reference = [("a", "1"), ("b", "1"), ("a", "1"), ("c", "9")]
    candidate = [("a", "1"), ("a", "1"), ("b", "1")]
    assert ex.match_animals(reference, candidate) == [0, 2, 1, None]


def _counts_to_animals(counts: dict[int, dict[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    dates, arms = [], []
    for arm, per_date in counts.items():
        for date, n in per_date.items():
            dates += [date] * n
            arms += [arm] * n
    return np.array(dates), np.array(arms)


_COUNTS = {
    0: dict.fromkeys(range(20210101, 20210113), 40),  # the wild type, on 12 dates
    1: {20210101: 10, 20210104: 12, 20210107: 9, 20210110: 11},
    2: {20210102: 15, 20210108: 15, 20210111: 14},
    3: {20210105: 20},  # one date only
    4: {20210103: 30, 20210106: 2},  # no subset of its dates holds 20-45%
}


def test_the_split_re_draws_identically() -> None:
    """The same dates and arms give the same split, every date on one side."""
    dates, arms = _counts_to_animals(_COUNTS)
    first = ex.draw_split(dates, arms, wild_type=0)
    second = ex.draw_split(dates[::-1], arms[::-1], wild_type=0)
    assert first["sides"] == second["sides"]
    assert set(first["sides"]) == {str(d) for d in np.unique(dates)}
    assert set(first["sides"].values()) == {"bands", "held_out"}
    wild = first["arms"]["0"]
    assert ex.WILD_TYPE_SHARE[0] <= wild["held_out_share"] <= ex.WILD_TYPE_SHARE[1]
    for arm in ("1", "2"):
        assert ex.ARM_SHARE[0] <= first["arms"][arm]["held_out_share"] <= ex.ARM_SHARE[1]


def test_an_arm_with_one_date_has_no_held_out_cohort() -> None:
    """A single-date arm sits on the bands side and is recorded as such."""
    dates, arms = _counts_to_animals(_COUNTS)
    split = ex.draw_split(dates, arms, wild_type=0)
    assert split["arms"]["3"]["single_date"]
    assert split["sides"]["20210105"] == "bands"
    assert split["arms"]["3"]["held_out_animals"] == 0


def test_an_arm_whose_dates_cannot_meet_the_share() -> None:
    """A share-infeasible arm is recorded, and needs only one date on each side."""
    dates, arms = _counts_to_animals(_COUNTS)
    split = ex.draw_split(dates, arms, wild_type=0)
    arm = split["arms"]["4"]
    assert arm["share_infeasible"]
    assert arm["held_out_dates"] == 1
    assert arm["dates"] - arm["held_out_dates"] == 1


def test_no_draw_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no draw meets the constraints, the split stops and reports the closest one."""
    monkeypatch.setattr(ex, "SPLIT_DRAWS", 5)
    monkeypatch.setattr(ex, "WILD_TYPE_SHARE", (0.9, 0.95))
    dates, arms = _counts_to_animals(_COUNTS)
    with pytest.raises(RuntimeError, match="closest was k ="):
        ex.draw_split(dates, arms, wild_type=0)


def _ibw_v2(values: np.ndarray, order: Literal["<", ">"]) -> bytes:
    kinds = {np.dtype("f4"): 2, np.dtype("f8"): 4, np.dtype("i4"): 32}
    kind = kinds[values.dtype]
    header = bytearray(126)
    header[0:2] = struct.pack(order + "h", 2)
    header[16:18] = struct.pack(order + "h", kind)
    header[16 + 42 : 16 + 46] = struct.pack(order + "i", len(values))
    return bytes(header) + values.astype(values.dtype.newbyteorder(order)).tobytes()


def _ibw_v5(values: np.ndarray, order: Literal["<", ">"]) -> bytes:
    header = bytearray(384)
    header[0:2] = struct.pack(order + "h", 5)
    header[64 + 12 : 64 + 16] = struct.pack(order + "i", len(values))
    header[64 + 16 : 64 + 18] = struct.pack(order + "h", 32)
    return bytes(header) + values.astype(np.dtype("i4").newbyteorder(order)).tobytes()


@pytest.mark.parametrize("order", ["<", ">"])
def test_the_igor_reader_reads_both_versions_and_byte_orders(
    tmp_path: Path,
    order: Literal["<", ">"],
) -> None:
    """Version 2 float waves and version 5 integer waves read back exactly, in either byte order."""
    floats = np.array([0.5, -1.25, np.nan, 3.0], dtype="f8")
    path = tmp_path / "w2.ibw"
    path.write_bytes(_ibw_v2(floats, order))
    assert np.array_equal(ex.read_ibw(path), floats, equal_nan=True)
    ints = np.array([0, 1, 2, 3, 2], dtype="i4")
    path5 = tmp_path / "w5.ibw"
    path5.write_bytes(_ibw_v5(ints, order))
    assert np.array_equal(ex.read_ibw(path5), ints)


def test_one_second_samples_of_a_29_97_hz_recording() -> None:
    """Each second takes its nearest frame, never past the last."""
    frames = ex._seconds(301)
    assert frames[0] == 0
    assert frames[-1] <= 300
    assert np.all(np.abs(frames - np.arange(len(frames)) * ex.ROBERTS_FPS) <= 0.5)


def test_mean_sem_cells_parse() -> None:
    """The sheet's "m +/- s" cells, with or without its "(SEM)" note, and its empty cells."""
    assert ex.parse_mean_sem("1.97 +/- 0.11 (SEM)") == (1.97, 0.11)
    assert ex.parse_mean_sem("2.68 +/- 0.11") == (2.68, 0.11)
    assert ex.parse_mean_sem("N/A") is None
    assert ex.parse_mean_sem(None) is None
    with pytest.raises(ValueError, match="cannot read"):
        ex.parse_mean_sem("two")


def test_calhouns_scale_is_unstated() -> None:
    """Calhoun's tracks load in pixels, with the deposit's onset counts."""
    tracks = refs.load_search_tracks("calhoun2014")
    assert tracks.units == "px"
    assert tracks.x.shape == (47, 1800)
    assert int(tracks.flag("omega").sum()) == 709
    assert int(tracks.flag("pirouette").sum()) == 3966
    # Every omega onset is also flagged as a pirouette.
    assert not np.any(tracks.flag("omega") & ~tracks.flag("pirouette"))


def test_roberts_tracks_carry_cohorts_and_states() -> None:
    """Roberts's wild-type cohorts and dead worms load in mm, with states in their codes."""
    tracks = refs.load_search_tracks("roberts2016")
    assert tracks.units == "mm"
    assert set(tracks.cohort.tolist()) == {"C15", "C21", "C57", "C162", "C186", "dead"}
    assert tracks.state_names == ("pause", "reverse", "forward")
    assert tracks.viterbi_names == ("X", "R", "F", "Y")
    assert tracks.state is not None
    assert set(np.unique(tracks.state).tolist()) <= {-1, 0, 1, 2}
    dead = tracks.cohort == "dead"
    assert np.nanmax(np.abs(tracks.x[dead])) < 0.01


def test_lopezcruz_lacks_two_d37_mutants() -> None:
    """The sheet has pdfr-1, linked to its same-dates wild type, but no tax-4 or tph-1."""
    data = refs.load_lopezcruz()
    assert data["absent_d37_mutants"] == ["tax-4", "tph-1"]
    pdfr = next(r for r in data["rows"] if r["genotype"].startswith("pdfr-1"))
    control = next(r for r in data["rows"] if r["row"] == pdfr["control_row"])
    assert control["genotype"].lower().startswith("wt")
    assert control["dates"] == pdfr["dates"]
    assert len(pdfr["mean"]) == len(data["bins"]) == 22
