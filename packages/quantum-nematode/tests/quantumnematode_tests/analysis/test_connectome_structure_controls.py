"""Tests for the connectome-structure control harness (metric parse + verdict logic)."""

import sys
from pathlib import Path

# The analysis script lives in scripts/analysis/ and imports sibling helper modules, so put that
# directory on the path first. Locate it by walking up to the repo root (robust to this test's
# nesting depth) rather than a hardcoded parent index.
_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
sys.path.insert(0, str(_root / "scripts" / "analysis"))

import connectome_structure_controls as csc  # noqa: E402  # pyright: ignore[reportMissingImports]


def _out_file(tmp_path: Path, statuses: list[str]) -> Path:
    """Write a synthetic run .out: one plateau-metric Run line per episode."""
    lines = [
        f"Run: {i}   Status: {s:<7} Reason: predator_evasion Steps: 2400    Eaten: "
        f"{10 if s == 'SUCCESS' else 3}/10  "
        for i, s in enumerate(statuses, 1)
    ]
    out = tmp_path / "run.out"
    out.write_text("\n".join(lines))
    return out


def test_success_reads_plateau_tail(tmp_path):
    """_success returns the final-quarter full-clear success % from the .out."""
    out = _out_file(tmp_path, ["FAILED"] * 6 + ["SUCCESS"] * 2)  # final quarter (last 2) = 100%
    assert csc._success(out) == 100.0


def test_missing_file_is_none(tmp_path):
    """A missing run .out yields None (the seed is dropped, not counted as zero)."""
    assert csc._success(tmp_path / "nope.out") is None


def _arm(base: float) -> dict[int, float]:
    # 8 paired seeds with small variation - enough for a one-sided Wilcoxon to reach q < 0.05.
    return {s: base + 0.5 * ((s % 3) - 1) for s in range(1, 9)}


def test_verdict_specific_wiring():
    """Wild-type clearly beating its rewired-null across paired seeds -> specific-wiring."""
    arms = {"wild_type": _arm(60.0), "rewired_null": _arm(35.0)}
    out: dict = {}
    csc.analyse(arms, out)
    assert out["verdict"]["verdict"] == "specific_wiring"
    assert out["verdict"]["mean_delta"] > 0


def test_verdict_degree_statistics():
    """Wild-type indistinguishable from its rewired-null (CI spans 0) -> degree-statistics."""
    arms = {"wild_type": _arm(50.0), "rewired_null": _arm(50.0)}
    out: dict = {}
    csc.analyse(arms, out)
    assert out["verdict"]["verdict"] == "degree_statistics"


def test_verdict_rewired_beats_is_reachable():
    """A significant rewired-WIN (CI entirely below zero) is labelled, not lost to 'inconclusive'.

    The reused Wilcoxon is one-sided (H1: wild > rewired), so this direction is detected via the
    bootstrap CI (ci_hi < 0), never via q.
    """
    arms = {"wild_type": _arm(35.0), "rewired_null": _arm(60.0)}
    out: dict = {}
    csc.analyse(arms, out)
    assert out["verdict"]["verdict"] == "rewired_beats_wildtype"
    assert out["verdict"]["ci_hi"] < 0


def test_verdict_insufficient_seeds():
    """Fewer than two common seeds -> the test cannot run."""
    arms = {"wild_type": {1: 60.0}, "rewired_null": {2: 35.0}}  # no common seed
    out: dict = {}
    csc.analyse(arms, out)
    assert out["verdict"]["verdict"] == "insufficient_seeds"


def test_load_parses_manifest(tmp_path):
    """load() maps <arm> <seed> <out> triples to {arm: {seed: success}}, dropping unparseable."""
    a = _out_file(tmp_path, ["SUCCESS"] * 8)
    manifest = tmp_path / "_manifest.txt"
    manifest.write_text(f"wild_type 1 {a}\nrewired_null 1 {a}\n# comment\n")
    # load() joins REPO / out_path; an absolute out_path makes that join a no-op (pathlib).
    arms = csc.load(manifest)
    assert arms["wild_type"][1] == 100.0
    assert arms["rewired_null"][1] == 100.0
