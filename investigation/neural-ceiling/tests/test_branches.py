"""Section 9's branch mapper on hand-made inputs: every cell, and the precedence rules."""

import copy

import pytest

from ceiling import readings as R

M = 0.02


def cell(diff, lo, hi):
    return {"diff": diff, "ci": (lo, hi)}


def fam(rank=1, diff=0.05, lo=0.01, hi=0.09, readable=19):
    return {"draws": 19, "readable_draws": readable, "rank": rank, "n_ranked": 1 + readable,
            "diff": diff, "ci": (lo, hi), "median": 0.1}


PASS, FAIL = fam(), fam(rank=3, diff=0.01, lo=-0.01, hi=0.03)


def base():
    held = {"m": M, "masked_readable": True,
            "R-A": {"masked": cell(0.05, 0.01, 0.09), "tied": cell(0.05, 0.01, 0.09)},
            "R-B": {"null-D": copy.deepcopy(PASS), "null-R": copy.deepcopy(PASS)},
            "R-D": cell(0.0, -0.05, 0.05)}
    return {
        "oracles": {"K": {"stimulations": True, "neurons": True},
                    "P": {"stimulations": True, "neurons": True, "worms": True},
                    "N": {"stimulations": False, "neurons": False}},
        "splits": {"stimulations": copy.deepcopy(held), "neurons": copy.deepcopy(held),
                   "worms": {"m": M, "masked_readable": True,
                             "R-C": {"tied": cell(0.05, 0.01, 0.09), "sparse-ER": cell(0.05, 0.01, 0.09),
                                     "denoising": cell(0.05, 0.01, 0.09)}}},
    }


def stim(result):
    return R.map_branches(result)["stimulations"]


# ---- R-A ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c, name", [
    (cell(-0.05, -0.09, -0.01), R.RA_BELOW),
    (cell(0.01, -0.01, 0.03), R.RA_NOT_ABOVE),
    (cell(0.01, 0.001, 0.02), R.RA_ABOVE_UNDER_M),
    (cell(0.03, 0.01, 0.05), R.RA_ABOVE),
    (cell(0.0, 0.0, 0.04), R.RA_NOT_ABOVE),  # a bound at exactly 0 includes 0
])
def test_ra_cells(c, name):
    r = base()
    r["splits"]["stimulations"]["R-A"]["masked"] = c
    r["splits"]["stimulations"]["R-A"]["tied"] = c
    out = stim(r)
    assert out["R-A"] == {"masked": name, "tied": name}


def test_ra_unreadable_tied():
    r = base()
    r["splits"]["stimulations"]["R-A"]["tied"] = None
    assert stim(r)["R-A"]["tied"].startswith(R.UNREADABLE)


# ---- gates ----------------------------------------------------------------------------------------


def test_k_failing_on_stimulations_gates_both_held_out_splits():
    r = base()
    r["oracles"]["K"]["stimulations"] = False
    out = R.map_branches(r)
    assert out["stimulations"] == {"R-A": {"masked": R.GATE_K, "tied": R.GATE_K},
                                   "R-B": R.GATE_K, "R-D": R.GATE_K}
    assert out["neurons"]["R-B"] == R.NOT_READ_K_STIM
    assert out["worms"]["R-C"] == R.RC_TABLE[(True, True)]  # held-out worms is still read


def test_k_failing_on_neurons_only():
    r = base()
    r["oracles"]["K"]["neurons"] = False
    out = R.map_branches(r)
    assert out["neurons"]["R-B"] == R.GATE_K
    assert out["stimulations"]["R-B"] == R.RB_TABLE[(True, True)]


def test_p_failing_makes_the_split_unreadable():
    r = base()
    r["oracles"]["P"]["stimulations"] = False
    out = stim(r)
    assert out["R-B"] == R.UNREADABLE
    assert out["R-A"]["masked"] == R.GATE_P and out["R-D"] == R.GATE_P


def test_gates_need_no_split_data():
    r = base()
    r["oracles"]["K"]["stimulations"] = False
    del r["splits"]["stimulations"], r["splits"]["neurons"]
    out = R.map_branches(r)
    assert out["stimulations"]["R-B"] == R.GATE_K


def test_description_only_split():
    r = base()
    r["description_only"] = ["neurons"]
    out = R.map_branches(r)
    assert out["neurons"]["R-B"] == R.DESCRIPTION
    assert out["disagreements"] == []  # a split that was not read cannot disagree


def test_a_gated_split_is_not_a_disagreement():
    r = base()
    r["oracles"]["P"]["neurons"] = False
    assert R.map_branches(r)["disagreements"] == []


# ---- R-B ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d, rr, name", [
    (PASS, PASS, R.RB_TABLE[(True, True)]),
    (PASS, FAIL, R.RB_TABLE[(True, False)]),
    (FAIL, PASS, R.RB_TABLE[(False, True)]),
    (FAIL, FAIL, R.RB_TABLE[(False, False)]),
])
def test_rb_table(d, rr, name):
    r = base()
    r["splits"]["stimulations"]["R-B"] = {"null-D": d, "null-R": rr}
    assert stim(r)["R-B"] == name


def test_rb_registered_names():
    assert R.RB_TABLE[(True, True)] == "Anatomy helps beyond degree and reciprocity statistics."
    assert R.RB_TABLE[(False, False)] == "Anatomy does not help a learned model here."


@pytest.mark.parametrize("bad", [
    fam(rank=2),  # not first of 20
    fam(diff=0.019),  # under m
    fam(lo=0.0),  # interval touches 0
])
def test_rb_pass_needs_all_three(bad):
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-D"] = bad
    assert stim(r)["R-B"] == R.RB_TABLE[(False, True)]


def test_rb_unreadable_when_masked_is():
    r = base()
    r["splits"]["stimulations"]["masked_readable"] = False
    r["oracles"]["N"]["stimulations"] = True  # unreadable takes precedence over biased
    assert stim(r)["R-B"] == R.UNREADABLE


def test_rb_structurally_biased_takes_precedence():
    r = base()
    r["oracles"]["N"]["stimulations"] = True
    r["splits"]["stimulations"]["R-B"]["null-D"] = fam(diff=-0.1, lo=-0.2, hi=-0.05)  # also reversed
    assert stim(r)["R-B"] == R.BIASED


def test_rb_reversed():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-R"] = fam(rank=15, diff=-0.05, lo=-0.08, hi=-0.03)
    assert stim(r)["R-B"] == R.REVERSED


def test_rb_reversal_needs_the_whole_interval_below_minus_m():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-R"] = fam(rank=15, diff=-0.05, lo=-0.08, hi=-0.019)
    assert stim(r)["R-B"] == R.RB_TABLE[(True, False)]


def test_rb_one_family_unreadable_is_read_alone():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-R"] = fam(readable=16)
    assert stim(r)["R-B"] == "null-R unreadable; null-D read alone: passes"
    r["splits"]["stimulations"]["R-B"]["null-D"] = fam(readable=17, rank=2)
    assert stim(r)["R-B"] == "null-R unreadable; null-D read alone: fails"


def test_rb_unreadable_family_is_not_checked_for_reversal():
    r = base()
    r["splits"]["stimulations"]["R-B"]["null-R"] = fam(readable=10, diff=-0.1, lo=-0.2, hi=-0.05)
    assert stim(r)["R-B"].endswith("passes")


# ---- R-C and denoising ----------------------------------------------------------------------------


@pytest.mark.parametrize("tied, er, name", [
    (True, True, "Anatomy helps at equal size, beyond random sparsity."),
    (False, True, R.RC_TABLE[(False, True)]),
    (True, False, R.RC_TABLE[(True, False)]),
    (False, False, "Neither the constraint nor sparsity helps here."),
])
def test_rc_table(tied, er, name):
    r = base()
    win, lose = cell(0.05, 0.01, 0.09), cell(0.01, -0.01, 0.03)
    r["splits"]["worms"]["R-C"]["tied"] = win if tied else lose
    r["splits"]["worms"]["R-C"]["sparse-ER"] = win if er else lose
    assert R.map_branches(r)["worms"]["R-C"] == name


def test_rc_unreadable_and_reversed():
    r = base()
    r["splits"]["worms"]["R-C"]["sparse-ER"] = cell(-0.05, -0.08, -0.03)
    assert R.map_branches(r)["worms"]["R-C"] == R.REVERSED
    r["oracles"]["P"]["worms"] = False
    out = R.map_branches(r)["worms"]
    assert out["R-C"] == R.UNREADABLE and out["denoising"] == R.UNREADABLE


def test_rc_win_needs_m():
    r = base()
    r["splits"]["worms"]["R-C"]["tied"] = cell(0.019, 0.005, 0.03)
    assert R.map_branches(r)["worms"]["R-C"] == R.RC_TABLE[(False, True)]


def test_denoising():
    r = base()
    assert R.map_branches(r)["worms"]["denoising"] == R.DENOISES
    r["splits"]["worms"]["R-C"]["denoising"] = cell(0.01, 0.001, 0.02)
    assert R.map_branches(r)["worms"]["denoising"] == R.NOT_DENOISES


# ---- R-D ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c, name", [
    (cell(0.05, 0.01, 0.09), R.RD_ABOVE),
    (cell(0.019, -0.01, 0.05), R.RD_DATA),
    (cell(-0.02, -0.06, 0.02), R.RD_DATA),  # exactly m below counts as within m
    (cell(-0.05, -0.09, -0.01), R.RD_MODEL),
    (cell(-0.05, -0.09, 0.01), R.RD_UNRESOLVED),
    (cell(0.05, -0.01, 0.09), R.RD_GAP),
])
def test_rd_cells(c, name):
    r = base()
    r["splits"]["stimulations"]["R-D"] = c
    assert stim(r)["R-D"] == name


# ---- precedence across splits ---------------------------------------------------------------------


def test_stimulations_govern_and_disagreements_are_named():
    r = base()
    r["splits"]["neurons"]["R-B"]["null-D"] = FAIL
    r["splits"]["neurons"]["R-A"]["tied"] = cell(-0.05, -0.09, -0.01)
    out = R.map_branches(r)
    assert out["governing"] == "stimulations"
    assert "R-B" in out["disagreements"] and "R-A/tied" in out["disagreements"]
    assert "R-D" not in out["disagreements"]
