"""Tests for the behavioural-chemotaxis validation harness (manifest parse + end-to-end verdict)."""

import json
import math
import sys
from pathlib import Path

# The analysis script lives in scripts/analysis/ and imports the package, so put that directory on
# the path first (walk up to the repo root, robust to this test's nesting depth).
_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "scripts" / "analysis").is_dir():
    _root = _root.parent
_analysis_dir = _root / "scripts" / "analysis"
if not _analysis_dir.is_dir():
    msg = f"could not locate scripts/analysis walking up from {Path(__file__).resolve()}"
    raise RuntimeError(msg)
sys.path.insert(0, str(_analysis_dir))

import behavioural_chemotaxis_validation as bcv  # noqa: E402  # pyright: ignore[reportMissingImports]


def _step(i: int, heading: float, dc_dt: float, grad_dir: float = 0.0) -> dict:
    """One serialised BehaviourStep (positions advance one unit along the heading)."""
    return {
        "step": i,
        "x": math.cos(heading) * i,
        "y": math.sin(heading) * i,
        "heading_rad": heading,
        "concentration": 0.5,
        "dc_dt": dc_dt,
        "grad_dir": grad_dir,
        "grad_strength": 1.0,
    }


def _klinokinesis_steps() -> list[dict]:
    """Frequent sharp turns down-gradient, sparse ones up-gradient -> a finite ratio > 1."""
    steps, h = [], 0.0
    for t in range(24):
        down = t % 2 == 0  # even steps head down-gradient (dc_dt < 0)
        # Always turn sharply down-gradient; up-gradient turn only occasionally (sparse, not zero,
        # so the down/up ratio stays finite and well above 1).
        turn = down or (t % 6 == 3)
        steps.append(_step(t, h, -1.0 if down else 1.0))
        h += 2.0 if turn else 0.05
    steps.append(_step(24, h, 0.0))
    return steps


def _capture_file(tmp_path: Path, seed: int, steps: list[dict]) -> Path:
    path = tmp_path / f"behaviour_seed{seed}.json"
    path.write_text(json.dumps({"runs": [{"run": 1, "seed": seed, "steps": steps}]}))
    return path


def _manifest(tmp_path: Path, entries: list[tuple[int, Path]]) -> Path:
    # Absolute capture paths: the harness resolves each as ``REPO / <path>``, which pathlib
    # short-circuits to the absolute path (so tmp captures outside the repo still resolve).
    manifest = tmp_path / "_manifest.txt"
    manifest.write_text(
        "# seed file\n" + "\n".join(f"{seed} {path}" for seed, path in entries),
    )
    return manifest


def test_load_manifest_skips_comments_and_missing(tmp_path, capsys):
    """Blank/comment lines are skipped and a missing capture file is dropped with a warning."""
    good = _capture_file(tmp_path, 42, _klinokinesis_steps())
    manifest = tmp_path / "_manifest.txt"
    manifest.write_text(
        f"# a comment\n\n42 {good}\n43 {tmp_path / 'nope.json'}\n",
    )
    seeds = bcv.load_manifest(manifest)
    assert set(seeds) == {42}
    assert "not found" in capsys.readouterr().out


def test_end_to_end_klinokinesis_reproduced(tmp_path):
    """Down-gradient-turning worms across seeds -> a klinokinesis ratio graded above the null."""
    entries = [(s, _capture_file(tmp_path, s, _klinokinesis_steps())) for s in range(42, 50)]
    manifest = _manifest(tmp_path, entries)
    seeds = bcv.load_manifest(manifest)
    theta = bcv._resolve_theta_sharp(seeds, theta_sharp=1.0, theta_percentile=85.0)
    summary = bcv.analyse(seeds, theta)
    assert summary["n_seeds"] == 8
    assert summary["klinokinesis"]["statistic"] == "down_up_turn_ratio"
    assert summary["klinokinesis"]["mean"] > summary["klinokinesis"]["null_value"]
    assert summary["klinokinesis"]["verdict"] in {"REPRODUCED", "PARTIAL"}


def test_figures_written(tmp_path):
    """--figure-dir emits both bias-curve PNGs."""
    entries = [(s, _capture_file(tmp_path, s, _klinokinesis_steps())) for s in range(42, 46)]
    seeds = bcv.load_manifest(_manifest(tmp_path, entries))
    theta = bcv._resolve_theta_sharp(seeds, theta_sharp=1.0, theta_percentile=85.0)
    summary = bcv.analyse(seeds, theta)
    figure_dir = tmp_path / "figures"
    bcv._write_figures(seeds, theta, summary, figure_dir)
    assert (figure_dir / "turn_rate_vs_dcdt.png").stat().st_size > 0
    assert (figure_dir / "curving_rate_vs_bearing.png").stat().st_size > 0
