"""Checks against the package's own build path (opt in: NC_PACKAGE_TESTS=1; about a minute).

They build the encoded brains as H0 and M-a built them, read T (never S), and fit nothing.
"""

import json
import os

import numpy as np
import pytest

from ceiling import config as C
from ceiling import inputs, mechanistic, supports
from ceiling.bootstrap import r_point

pytestmark = [pytest.mark.package,
              pytest.mark.skipif(os.environ.get("NC_PACKAGE_TESTS") != "1", reason="NC_PACKAGE_TESTS=1")]


def test_anatomical_union_is_the_wild_type_brains_support():
    s = supports.anatomical()
    assert s.size == 5103
    assert s.keys() == supports.brain_support(None).keys()


def test_null_d_supports_match_the_built_brains():
    """Counts from rewire_degree_preserving at each seed; supports asserted equal to H0's brains."""
    sizes = {}
    for seed in C.SEEDS_NULL_D:
        sizes[seed] = supports.null_d(seed).size  # raises if the supports differ
    # The registration describes them as 5,729 to 5,749 (about 12% above masked's 5,103); the
    # 19 built brains give 5,719 to 5,754 (seeds 9741 and 9742 give exactly 5,749 and 5,729).
    assert min(sizes.values()) == 5719 and max(sizes.values()) == 5754, sizes
    assert 1.11 < np.median(list(sizes.values())) / 5103 < 1.13


def test_null_r_on_the_anatomical_support():
    base = supports.anatomical()
    out = supports.null_r(base, C.SEEDS_NULL_R[0])
    assert out.size == base.size and out.reciprocal_entries() == base.reciprocal_entries() == 3168
    assert out.report["fraction_moved"] >= C.NULL_R_MIN_MOVED


def test_regenerated_responses_reproduce_m_a_on_T():
    """M-a's committed r against the training matrix: box centre 0.0898, C.2 0.0725."""
    names = inputs.load_names()
    resp = mechanistic.responses(names)
    assert max(resp["parity"].values()) <= mechanistic.PARITY_TOL
    T = inputs.load_T()
    ma = json.loads((C.REPO / "investigation/spike/ma_c2/result.json").read_text())["readings"]
    for name, key in (("H0", "box_centre/start/1"), ("C.2", "c2/start/1")):
        M = resp[name].copy()
        np.fill_diagonal(M, np.nan)
        assert r_point(T, M) == pytest.approx(ma[key]["train"]["r"], abs=1e-6)
