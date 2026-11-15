"""Every stage in its registered order on a tiny synthetic world: reduced families, two lambdas,
two iterations per fit, and the scorer on a temporary repository with a synthetic matrix file.
It checks the wiring (files, reads, branches), not any number."""

import json

import numpy as np
import pytest

from ceiling import config as C
from ceiling import inputs, pipeline, stages, supports
from ceiling import plan as P
from ceiling.scorer import Scorer

from test_guard import REL, repo, write_stams  # noqa: F401 - the fixture

N = 160


@pytest.fixture
def world(tmp_path, monkeypatch, repo):  # noqa: F811
    small = {"NULL_D": ["nullD:9741", "nullD:9742"], "NULL_R": ["nullR:9781", "nullR:9782"],
             "ER": ["er:9801"], "PRIOR": ["prior:masked", "prior:nullD:9741"], "SWEEP": ["tied:2"],
             "SELECTION_SET": ["masked", "nullD:9741", "nullR:9781"]}
    for name, value in small.items():
        monkeypatch.setattr(P, name, value)
    monkeypatch.setattr(stages, "FAMILY_NAMES", {"null-D": small["NULL_D"], "null-R": small["NULL_R"],
                                                 "sparse-ER": small["ER"]})
    for name, value in {"SEEDS_P": (9773,), "N_TRUTH_SEEDS": (9741,), "MAX_ITER": 2, "MAX_ITER_K": 2,
                        "LAMBDA_GRID": (1e-3, 1e-2), "NULL_READABLE_MIN": 1, "P_MIN_DRAWS": 1,
                        "N_FAIL_MIN_TRUTHS": 1, "BUSY_PATTERN": "no-such-process-9771x"}.items():
        monkeypatch.setattr(C, name, value)
    stams = tmp_path / "stams.npz"
    sha = write_stams(stams)
    monkeypatch.setattr(stages, "make_scorer", lambda confirm, amendment=None: Scorer(
        confirm, amendment, repo=repo, registration=repo / REL, reads=repo / REL.parent / "READS.md",
        stams=stams, stams_sha256=sha))
    run = tmp_path / "run"
    run.mkdir()
    classes = [f"c{k % 74:02d}" for k in range(C.N_MEASURED)]
    (run / "layout.json").write_text(json.dumps({"names": classes, "classes": classes}))
    names = ["masked"] + small["NULL_D"] + small["NULL_R"] + small["ER"]
    built = {name: supports.sparse_er(400, N, 9900 + k) for k, name in enumerate(names)}
    rng = np.random.default_rng(0)
    for s in built.values():
        s.chem, s.gap = rng.integers(0, 4, s.size).astype(float), rng.integers(0, 2, s.size).astype(float)
    pipeline.save_supports(run, built, np.arange(C.N_MEASURED))
    np.savez(run / "mechanistic.npz", H0=rng.normal(size=(154, 154)), C2=rng.normal(size=(154, 154)))
    pipeline.save_dataset(run, "real", inputs.load_T(stams, sha))
    pipeline._context.cache_clear()
    pipeline._dataset.cache_clear()
    yield run, repo
    pipeline._context.cache_clear()
    pipeline._dataset.cache_clear()


def test_every_stage_in_order(world):
    run, repo_dir = world
    pre = stages.preflight(run, True, workers=0)
    assert pre["projection"]["action"] == "proceed" and pre["cuts"] == []
    assert pre["timings"]["tied"]["both_move"] in (True, False)
    assert set(pre["c2_minus_h0_against_T"]) == set(C.SPLITS)
    orc = stages.oracles_stage(run, True, workers=0)
    assert set(orc["gates"]) == {"K", "P", "N"}
    assert set(orc["r_att"]) == set(C.HELD_OUT_SPLITS)
    real = stages.real_stage(run, workers=0)
    assert set(real["splits"]) <= set(C.SPLITS)
    results = stages.score_stage(run, True)
    assert set(results["branches"]) >= {"stimulations", "neurons", "worms", "governing"}
    assert (run / "results.json").exists() and (run / "results_per_arm.csv").exists()
    log = (repo_dir / REL.parent / "READS.md").read_text()
    assert [line for line in log.splitlines() if line.startswith("- quantity: ")] == [
        "- quantity: `entry_sets`", "- quantity: `section3`", "- quantity: `registered_scores`"]
    # a rerun resumes from the checkpoints and refuses the second registered reading
    stages.real_stage(run, workers=0)
    with pytest.raises(Exception, match="already read"):
        stages.score_stage(run, True)
