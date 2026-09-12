"""Tests for the V.2 graph-property probe: the measures, and the constant-property reporting."""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
_analysis_dir = _root / "scripts" / "analysis"
if not _analysis_dir.is_dir():  # fail fast rather than an opaque ModuleNotFoundError later
    msg = f"could not locate scripts/analysis walking up from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_analysis_dir))

import wiring_premise_graph_probe as probe  # noqa: E402  # pyright: ignore[reportMissingImports]


def test_bfs_depths_counts_hops_from_the_nearest_source():
    """Hop counts are breadth-first from the source set, not per-source."""
    adjacency = {"a": ["b"], "b": ["c"], "c": ["d"], "x": ["c"]}
    depths = probe._bfs_depths(adjacency, {"a"})
    assert depths == {"a": 0, "b": 1, "c": 2, "d": 3}

    # With two sources the nearer one wins.
    depths_two = probe._bfs_depths(adjacency, {"a", "x"})
    assert depths_two["c"] == 1
    assert depths_two["d"] == 2


def test_bfs_depths_ignores_unreachable_nodes():
    """A node with no inbound path is absent rather than infinite."""
    depths = probe._bfs_depths({"a": ["b"]}, {"a"})
    assert "z" not in depths


def test_char_path_length_averages_over_reachable_ordered_pairs():
    """A directed chain a->b->c: pairs (a,b)=1, (a,c)=2, (b,c)=1, mean 4/3."""
    adjacency = {"a": ["b"], "b": ["c"]}
    assert probe._char_path_length(adjacency, ["a", "b", "c"]) == pytest.approx(4 / 3)


def test_bh_fdr_matches_a_known_step_up():
    """The correction is Benjamini-Hochberg step-up, monotone in the sorted p-values."""
    qs = probe._bh_fdr([0.01, 0.02, 0.03, 0.04])
    assert qs == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert all(a <= b + 1e-12 for a, b in pairwise(qs))


def test_a_constant_property_is_reported_as_constant_not_as_a_failed_correlation(monkeypatch):
    """A property with no variance across rewirings says so, and carries no rho or q.

    Three of the four registered properties turned out constant, and reporting them as failed
    correlations would have hidden the reason they cannot discriminate.
    """
    wild = dict.fromkeys(probe.PROPERTIES, 1.0)
    monkeypatch.setattr(probe, "load_cook_2019_hermaphrodite", lambda: "wild")
    monkeypatch.setattr(probe, "rewire_degree_preserving", lambda _c, _rng: "rewired")
    monkeypatch.setattr(
        probe,
        "graph_properties",
        lambda graph: wild if graph == "wild" else dict.fromkeys(probe.PROPERTIES, 7.0),
    )
    monkeypatch.setattr(probe, "SEEDS", range(1, 9))
    monkeypatch.setattr(
        probe,
        "load_times",
        lambda _panels: {s: float(s) * 10 for s in range(1, 9)},
    )

    report = probe.analyse([Path("unused.json")])
    for prop in probe.PROPERTIES:
        entry = report["correlations"][prop]
        assert entry["constant_at"] == 7.0
        assert np.isnan(entry["rho"])
        assert "cannot discriminate" in entry["note"]


def test_load_times_rejects_a_seed_present_in_two_panels(tmp_path):
    """A seed scored twice would weight it twice; that raises rather than overwriting."""
    payload = (
        '{"efficiency": {"thermal": {"per_seed": {"rewired_null": {"1": {"'
        + probe.METRIC
        + '": 100.0}}}}}}'
    )
    one, two = tmp_path / "a.json", tmp_path / "b.json"
    one.write_text(payload)
    two.write_text(payload)
    assert probe.load_times([one]) == {1: 100.0}
    with pytest.raises(ValueError, match="more than one panel"):
        probe.load_times([one, two])
