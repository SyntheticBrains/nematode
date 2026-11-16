"""Boundary probes of the mapper (ported from the independent review) and amendment 1's gates
(item 6) and R-C comparators (item 7)."""

import pytest

from ceiling import config as C
from ceiling import readings as R

from test_branches import base, cell, fam, stim

M = 0.02


def test_ra_diff_exactly_m_is_above():
    r = base()
    r["splits"]["stimulations"]["R-A"]["masked"] = cell(M, 0.001, 0.05)
    assert stim(r)["R-A"]["masked"] == R.RA_ABOVE


def test_ra_upper_bound_exactly_zero_includes_zero():
    r = base()
    r["splits"]["stimulations"]["R-A"]["masked"] = cell(-0.01, -0.03, 0.0)
    assert stim(r)["R-A"]["masked"] == R.RA_NOT_ABOVE


def test_rb_diff_exactly_m_passes():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-D"] = fam(diff=M)
    assert stim(r)["R-B"] == R.RB_TABLE[(True, True)]


def test_rb_upper_bound_exactly_minus_m_is_not_reversed():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-R"] = fam(rank=12, diff=-0.05, lo=-0.08, hi=-M)
    assert stim(r)["R-B"] == R.RB_TABLE[(True, False)]


@pytest.mark.parametrize("readable, live", [(16, False), (17, True)])
def test_rb_16_vs_17(readable, live):
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-R"] = fam(readable=readable)
    assert (stim(r)["R-B"] == R.RB_TABLE[(True, True)]) == live


def test_rb_both_unreadable():
    r = base()
    r["splits"]["stimulations"]["R-B"] = {"null-D": fam(readable=16), "null-R": fam(readable=3)}
    assert stim(r)["R-B"].startswith(R.UNREADABLE)


def test_rb_reversed_beats_a_passing_family():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-D"] = fam(rank=19, diff=-0.1, lo=-0.2, hi=-0.05)
    assert stim(r)["R-B"] == R.REVERSED


def test_rd_exactly_m_above_is_data_limited():
    r = base()
    r["splits"]["stimulations"]["R-D"] = cell(M, 0.001, 0.05)
    assert stim(r)["R-D"] == R.RD_DATA


def test_rd_gap_and_denoising_labels_are_the_registered_ones():
    """Amendment 1 item 13 registers IMPLEMENTATION_NOTES item 21's labels."""
    r = base()
    r["splits"]["stimulations"]["R-D"] = cell(0.05, -0.01, 0.09)
    assert stim(r)["R-D"] == "above, unresolved"
    r["splits"]["worms"]["R-C"]["denoising"] = cell(0.01, 0.001, 0.02)
    assert R.map_branches(r)["worms"]["denoising"] == "does not denoise beyond copying"


# ---- amendment 1, item 6: gates ---------------------------------------------------------------------


def test_k_gates_r_a_and_n_does_not():
    r = base()
    r["oracles"]["N"]["stimulations"] = True
    out = stim(r)
    assert out["R-A"]["masked"] == R.RA_ABOVE and out["R-B"] == R.BIASED and out["R-D"] == R.RD_DATA
    r["oracles"]["K"]["stimulations"] = False
    assert stim(r)["R-A"]["masked"] == R.GATE_K


def test_p_worm_line_gates_r_c_and_denoising_only():
    r = base()
    r["oracles"]["P"]["worms"] = False
    out = R.map_branches(r)
    assert out["worms"] == {"R-C": R.UNREADABLE, "denoising": R.UNREADABLE}
    assert out["stimulations"]["R-B"] == R.RB_TABLE[(True, True)]


def test_an_unreadable_masked_is_reported_not_refused():
    r = base()
    s = r["splits"]["stimulations"]
    s.update(masked_readable=False, m=float("nan"))
    s["R-A"]["masked"] = None
    s["R-D"] = None
    out = stim(r)
    assert out["R-A"]["masked"] == R.ARM_UNREADABLE
    assert out["R-A"]["tied"] == R.M_UNDEFINED  # m needs masked's r
    assert out["R-B"] == R.UNREADABLE and out["R-D"] == R.ARM_UNREADABLE
    w = r["splits"]["worms"]
    w.update(masked_readable=False)
    worms = R.map_branches(r)["worms"]
    assert worms["R-C"] == R.UNREADABLE and worms["denoising"] == R.UNREADABLE


# ---- amendment 1, item 7: R-C comparators -------------------------------------------------------------


def test_rc_unreadable_tied_is_unreadable_not_fails():
    r = base()
    r["splits"]["worms"]["R-C"]["tied"] = None
    assert R.map_branches(r)["worms"]["R-C"] == "sparse-ER read alone: wins"


def test_rc_unreadable_sparse_er_is_unreadable_not_fails():
    r = base()
    r["splits"]["worms"]["R-C"]["sparse-ER"] = None
    r["splits"]["worms"]["R-C"]["tied"] = cell(0.01, -0.01, 0.03)
    assert R.map_branches(r)["worms"]["R-C"] == "tied read alone: fails"


def test_rc_both_comparators_unreadable():
    r = base()
    r["splits"]["worms"]["R-C"].update(tied=None, **{"sparse-ER": None})
    assert R.map_branches(r)["worms"]["R-C"].startswith(R.UNREADABLE)


def test_rc_reversal_is_read_on_the_readable_contrast():
    r = base()
    r["splits"]["worms"]["R-C"].update(tied=None, **{"sparse-ER": cell(-0.05, -0.08, -0.03)})
    assert R.map_branches(r)["worms"]["R-C"] == R.REVERSED


@pytest.mark.parametrize("readable, ok", [(3, False), (4, True), (5, True)])
def test_sparse_er_needs_four_of_five_readable(readable, ok):
    import numpy as np

    from ceiling import report

    boot = np.zeros(10)
    scores = {"masked": {"readable": True, "r": 0.3, "boot": boot + 0.3},
              "floor-C": {"readable": True, "r": 0.1, "boot": boot + 0.1},
              "tied": {"readable": True, "r": 0.2, "boot": boot + 0.2},
              "T": {"readable": True, "r": 0.15, "boot": boot + 0.15}}
    ers = [f"er:{9801 + k}" for k in range(5)]
    for k, name in enumerate(ers):
        scores[name] = {"readable": k < readable, "r": 0.1, "boot": boot + 0.1}
    rd = report.split_readings(scores, "worms", {"null-D": [], "null-R": [], "sparse-ER": ers}, ("floor-C",))
    assert (rd["R-C"]["sparse-ER"] is not None) == ok
    assert rd["R-C"]["sparse-ER_readable_draws"] == readable
    assert C.ER_READABLE_MIN == 4
