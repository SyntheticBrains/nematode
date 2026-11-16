"""Selection, extension, finals, pooling, checkpoint resume and the busy check, on a tiny synthetic
run directory (154 measured neurons, a small support over 160 nodes, synthetic T)."""

import json
import math
import subprocess
import sys
import time

import numpy as np
import pytest

from ceiling import config as C
from ceiling import pipeline, runner, splits, supports
from ceiling import plan as P

N = 160


@pytest.fixture
def run_dir(tmp_path):
    classes = [f"c{k % 74:02d}" for k in range(C.N_MEASURED)]
    (tmp_path / "layout.json").write_text(json.dumps({"names": classes, "classes": classes}))
    built = {name: supports.sparse_er(400, N, 9800 + k) for k, name in enumerate(P.SELECTION_SET)}
    pipeline.save_supports(tmp_path, built, np.arange(C.N_MEASURED))
    rng = np.random.default_rng(0)
    T = rng.normal(size=(C.N_MEASURED, C.N_MEASURED))
    T[rng.random(T.shape) < 0.1] = np.nan
    np.fill_diagonal(T, np.nan)
    pipeline.save_dataset(tmp_path, "real", T)
    pipeline._context.cache_clear()
    pipeline._dataset.cache_clear()
    return tmp_path


def test_execute_gives_validation_loss_and_held_out_prediction(run_dir):
    base = {"dataset": "real", "split": "stimulations", "fold": 0, "arm": "masked",
            "lam": 1e-2, "next_lam": 1e-1, "max_iter": 3}
    val = pipeline.execute(str(run_dir), {**base, "id": "a", "inner": 0})
    assert math.isfinite(val["val_loss"]) and "pred" not in val
    fin = pipeline.execute(str(run_dir), {**base, "id": "b", "inner": None})
    layout = splits.make_layout(json.loads((run_dir / "layout.json").read_text())["classes"])
    hidden = splits.hidden_mask("stimulations", layout.in_fold(0)) & splits.off_diagonal()
    assert fin["pred"].shape == (int(hidden.sum()),)
    assert fin["summary"]["n_params"] == 400


def stub(losses):
    """An execute that scores each lambda by a fixed table, with no fitting."""
    calls = []

    def execute(run_dir, task):
        calls.append(task["id"])
        out = {"id": task["id"], "summary": {"readable": True}}
        if task["inner"] is not None:
            out["val_loss"] = losses(task["lam"])
        else:
            out["pred"] = np.zeros(1)
        return out

    return execute, calls


def test_selection_picks_the_smallest_summed_loss(run_dir):
    execute, calls = stub(lambda lam: (math.log10(lam) + 3) ** 2)
    units = {("stimulations", 0): ["masked", "nullD:9741"]}
    choice = pipeline.select_lambdas(run_dir, "t", "real", units, execute, workers=0, check_busy=False)
    rec = choice["stimulations/0|sparse"]
    assert rec["lambda"] == 1e-3 and rec["extension"] is None
    assert len(calls) == len(P.SELECTION_SET) * len(C.LAMBDA_GRID) * C.N_INNER  # 135
    assert rec["losses"][1e-3] == 0.0


def test_an_edge_choice_is_extended_once(run_dir):
    execute, calls = stub(lambda lam: math.log10(lam))  # smaller is always better
    units = {("worms", None): ["masked"]}
    choice = pipeline.select_lambdas(run_dir, "t", "real", units, execute, workers=0, check_busy=False,
                                     selection_override={"sparse": ["masked"]})
    rec = choice["worms|sparse"]
    assert rec["extension"] == pytest.approx(1e-5)
    assert rec["lambda"] == pytest.approx(1e-5)  # chosen on the extended grid, not extended again
    assert len(calls) == (len(C.LAMBDA_GRID) + 1) * C.N_INNER


def test_checkpoint_resumes_without_refitting(run_dir):
    execute, calls = stub(lambda lam: 1.0)
    units = {("neurons", 1): ["masked"]}
    over = {"sparse": ["masked"]}
    pipeline.select_lambdas(run_dir, "r", "real", units, execute, workers=0, check_busy=False,
                            selection_override=over)
    first = len(calls)
    with open(run_dir / "checkpoints" / "r.jsonl", "a") as handle:
        handle.write('{"id": "torn')  # a line cut by a stop is ignored
    pipeline.select_lambdas(run_dir, "r", "real", units, execute, workers=0, check_busy=False,
                            selection_override=over)
    assert len(calls) == first


def test_pool_needs_every_fold(run_dir):
    T = np.load(run_dir / "datasets/real.npz")["T"]
    layout = pipeline._context(str(run_dir))["layout"]
    units = {("stimulations", f): ["masked"] for f in range(C.N_FOLDS)}
    choice = {f"stimulations/{f}|sparse": {"lambda": 1e-2, "extension": None} for f in range(C.N_FOLDS)}

    def execute_with(bad_fold):
        def execute(rd, task):
            hidden = pipeline.unit_masks(T, layout, task["split"], task["fold"])["hidden"]
            ok = task["fold"] != bad_fold
            return {"id": task["id"], "summary": {"readable": ok},
                    "pred": np.full(int(hidden.sum()), float(task["fold"])) if ok else None}
        return execute

    ck, tasks = pipeline.final_fits(run_dir, "p", "real", units, choice, execute_with(3), workers=0,
                                    check_busy=False)
    assert pipeline.pool(run_dir, ck, tasks, T, layout)["stimulations"]["masked"] is None
    ck, tasks = pipeline.final_fits(run_dir, "q", "real", units, choice, execute_with(-1), workers=0,
                                    check_busy=False)
    matrix = pipeline.pool(run_dir, ck, tasks, T, layout)["stimulations"]["masked"]
    for f in range(C.N_FOLDS):  # each column holds its own fold's prediction
        assert np.all(matrix[:, layout.in_fold(f)][~np.eye(C.N_MEASURED, dtype=bool)[:, layout.in_fold(f)]] == f)
    # a split with fewer folds than registered is unreadable
    one = {("stimulations", 0): ["masked"]}
    ck, tasks = pipeline.final_fits(run_dir, "s", "real", one, choice, execute_with(-1), workers=0,
                                    check_busy=False)
    assert pipeline.pool(run_dir, ck, tasks, T, layout)["stimulations"]["masked"] is None


def test_one_spawned_worker_runs_real_fits(run_dir, monkeypatch):
    monkeypatch.setattr(C, "MAX_ITER", 3)
    units = {("stimulations", 2): ["masked"]}
    choice = pipeline.select_lambdas(run_dir, "w", "real", units, workers=1, check_busy=False,
                                     selection_override={"sparse": ["masked"]})
    ck, tasks = pipeline.final_fits(run_dir, "w", "real", units, choice, workers=1, check_busy=False)
    (tid,) = tasks
    layout = pipeline._context(str(run_dir))["layout"]
    hidden = splits.hidden_mask("stimulations", layout.in_fold(2)) & splits.off_diagonal()
    pred = ck.prediction(tid)
    assert pred.shape == (int(hidden.sum()),) and np.isfinite(pred).all()
    inner = [r for k, r in ck.done.items() if k != tid]
    assert inner and all(r["summary"]["n_iter"] <= 3 and math.isfinite(r["val_loss"]) for r in inner)
    extended = choice["stimulations/2|sparse"]["extension"] is not None
    assert len(ck.done) == (len(C.LAMBDA_GRID) + extended) * C.N_INNER + 1


def test_busy_processes_refuse_the_run(run_dir, monkeypatch):
    marker = "9771.123"
    proc = subprocess.Popen([sys.executable, "-c", "import time, sys; time.sleep(30)", marker])
    try:
        time.sleep(0.2)
        monkeypatch.setattr(C, "BUSY_PATTERN", marker.replace(".", r"\."))
        assert any(marker in line for line in runner.busy_processes())
        with pytest.raises(runner.Busy):
            runner.run_tasks([{"id": "x"}], run_dir, "busy", lambda rd, t: t, workers=0)
    finally:
        proc.kill()
        proc.wait()
    monkeypatch.setattr(C, "BUSY_PATTERN", r"no-such-process-9771x")
    assert runner.busy_processes() == []


def test_planned_counts_match_the_registered_budget():
    counts = P.fit_counts()
    total = {k: sum(v.values()) for k, v in counts.items()}
    assert total["real"] == 11 * 330 + 16  # 330 per fold-unit, the untied arm on held-out worms
    assert all(total[f"P{s}"] == 1935 for s in C.SEEDS_P)
    assert all(total[f"N{s}"] == 1540 for s in C.N_TRUTH_SEEDS)
    cut = P.fit_counts(("description_arms", "neuron_split_oracles", "n_two_truths"))
    assert "N9743" not in cut and sum(cut["real"].values()) < total["real"]
    proj = P.cut_order({k: 1e-3 for k in ("sparse", "sparse_nullD", "prior", "K", "tied", "dense",
                                            P.UNTIED, *P.SWEEP)})
    assert proj["cuts"] == [] and proj["action"] == "proceed"
    slow = P.cut_order({k: 60.0 for k in ("sparse", "sparse_nullD", "prior", "K", "tied", "dense",
                                            P.UNTIED, *P.SWEEP)})
    assert slow["cuts"] == list(C.CUTS) and slow["action"] == "stop and amend"
