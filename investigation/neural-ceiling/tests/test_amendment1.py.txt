"""Amendment 1's items, and a test for every mutation the independent review found missed.

Synthetic data only; every fit is a few iterations except K's known answer (one fold, 2,000 cap).
"""

import json
import math

import numpy as np
import pytest
import torch

from ceiling import bootstrap, fitting, models, pipeline, readings, report, splits, stages, supports
from ceiling import config as C
from ceiling import plan as P

K = C.N_MEASURED
CLASSES = [f"c{k % 74:02d}" for k in range(K)]


def run_dir_with(tmp_path, T, built, constants=None, name="real"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "layout.json").write_text(json.dumps({"names": CLASSES, "classes": CLASSES}))
    pipeline.save_supports(tmp_path, built, np.arange(K))
    pipeline.save_dataset(tmp_path, name, T, constants)
    pipeline._context.cache_clear()
    pipeline._dataset.cache_clear()
    return tmp_path


# ---- item 1: oracle standardisation -----------------------------------------------------------------


def k_truth(n=160, nnz=900, sd=0.25, seed=3, mu=1.3, sigma=2.5):
    """A model-generated truth in standardised units, put into raw units by (mu, sigma)."""
    rng = np.random.default_rng(seed)
    s = supports.sparse_er(nnz, n, 100 + seed)
    with torch.no_grad():
        m_hat = models.SparseSeries(s.rows, s.cols, n, np.arange(K), torch.tensor(rng.normal(0, sd, nnz))).response().numpy()
    T = mu + sigma * m_hat
    T[rng.random((K, K)) < 0.08] = np.nan
    np.fill_diagonal(T, np.nan)
    return s, T, (mu, sigma)


def k_fit(tmp_path, s, T, constants):
    d = run_dir_with(tmp_path, T, {"masked": s}, constants, name="K")
    task = {"id": "k", "dataset": "K", "split": "stimulations", "fold": 0, "inner": None, "arm": "masked",
            "lam": C.K_LAMBDA, "next_lam": 1e-3, "max_iter": C.MAX_ITER_K}
    out = pipeline.execute(str(d), task)
    layout = splits.make_layout(CLASSES)
    hidden = pipeline.unit_masks(T, layout, "stimulations", 0)["hidden"]
    full = np.full((K, K), np.nan)
    full[hidden] = out["pred"]
    scored = hidden & np.isfinite(T) & ~layout.in_fold(0)[:, None]
    return out["summary"], float(np.corrcoef(full[scored], T[scored])[0, 1])


def test_k_reaches_its_known_answer_with_the_truth_constants(tmp_path):
    """Amendment 1, item 1: a K data set standardised with its truth's own (mean, SD) is in the
    model class again: visible loss below K's near-zero bar and held-out r at K's bar."""
    s, T, constants = k_truth()
    summary, r = k_fit(tmp_path / "truth", s, T, constants)
    assert summary["mu"] == constants[0] and summary["sigma"] == constants[1]
    assert summary["data_loss"] < C.K_NEAR_ZERO_LOSS
    assert r >= C.K_BAR


def test_own_constants_would_have_failed_k(tmp_path):
    """The same truth re-standardised by its own visible spread leaves the class."""
    s, T, _ = k_truth()
    summary, r = k_fit(tmp_path / "own", s, T, None)
    assert summary["data_loss"] > C.K_NEAR_ZERO_LOSS and r < C.K_BAR


def test_with_constants_standardises_by_the_given_pair():
    T = np.array([[np.nan, 3.0], [5.0, np.nan]])
    y, mu, sigma = fitting.with_constants(T, 1.0, 2.0)
    assert (mu, sigma) == (1.0, 2.0) and y[0, 1] == 1.0 and y[1, 0] == 2.0
    with pytest.raises(ValueError):
        fitting.with_constants(T, 0.0, 0.0)


# ---- the review's leak check, as a test -------------------------------------------------------------


def poisoned_world():
    rng = np.random.default_rng(0)
    n = 170
    truth_s = supports.sparse_er(500, n, 1)
    with torch.no_grad():
        M = models.SparseSeries(truth_s.rows, truth_s.cols, n, np.arange(K),
                                torch.tensor(rng.normal(0, .5, 500))).response().numpy()
    T = 2 + 3 * M + rng.normal(0, .3, (K, K))
    T[rng.random((K, K)) < .1] = np.nan
    np.fill_diagonal(T, np.nan)
    s_fin = rng.random((K, K)) > .4
    np.fill_diagonal(s_fin, False)
    mech = {"H0": rng.normal(size=(K, K)), "C.2": rng.normal(size=(K, K))}
    return rng, truth_s, T, s_fin, mech


@pytest.mark.parametrize("split, fold", [("stimulations", 0), ("neurons", 1)])
def test_no_fit_floor_or_calibration_reads_a_hidden_entry(tmp_path, split, fold):
    """Poison every hidden entry (values and finite pattern): final fits, inner-CV losses, floors
    and calibrations are unchanged on the scored entries."""
    rng, truth_s, T, s_fin, mech = poisoned_world()
    layout = splits.make_layout(CLASSES)
    in_fold = layout.in_fold(fold)
    hidden = splits.hidden_mask(split, in_fold) & splits.off_diagonal()
    poison = rng.normal(50, 20, (K, K))
    poison[rng.random((K, K)) < .3] = np.nan
    Tp = T.copy()
    Tp[hidden] = poison[hidden]
    rule = splits.entry_set_stimulations if split == "stimulations" else splits.entry_set_neurons
    entries = rule(np.isfinite(T), s_fin, in_fold)
    assert entries.sum() > 100
    got = {}
    for tag, Tm in (("clean", T), ("poison", Tp)):
        d = run_dir_with(tmp_path / tag, Tm, {"masked": truth_s})
        res = {}
        for arm in ("masked", "tied"):
            for inner in (None, 0):
                task = {"id": "x", "dataset": "real", "split": split, "fold": fold, "inner": inner,
                        "arm": arm, "lam": 1e-2, "next_lam": 1e-1, "max_iter": 4}
                out = pipeline.execute(str(d), task)
                if inner is None:
                    full = np.full((K, K), np.nan)
                    full[hidden] = out["pred"]
                    res[f"{arm}/final"] = full[entries]
                else:
                    res[f"{arm}/inner"] = out["val_loss"]
        refs, coefs = stages.reference_pool(Tm, layout, [(split, fold)], mech)
        for name, mat in refs[split].items():
            res[f"ref/{name}"] = mat[entries]
        res["coefs"] = json.dumps(coefs, default=float)
        got[tag] = res
    for key, clean in got["clean"].items():
        dirty = got["poison"][key]
        same = clean == dirty if isinstance(clean, (str, float)) else np.array_equal(clean, dirty, equal_nan=True)
        assert same, key


# ---- the bootstrap unit, m, the rank, the median, pairing, r_ref ------------------------------------


def test_the_bootstrap_unit_is_the_class():
    layout = splits.make_layout(CLASSES)
    rows, cols = np.array([0, 1, 2, 80]), np.array([5, 6, 77, 3])
    assert np.array_equal(splits.unit_class("stimulations", layout, rows, cols), layout.class_index[cols])
    assert np.array_equal(splits.unit_class("worms", layout, rows, cols), layout.class_index[cols])
    assert np.array_equal(splits.unit_class("neurons", layout, rows, cols), layout.class_index[rows])
    # through score_split: the resamples weight each entry by its stimulated class's count
    rng = np.random.default_rng(1)
    target = rng.normal(size=(K, K))
    pred = target + rng.normal(size=(K, K))
    entries = rng.random((K, K)) < 0.3
    np.fill_diagonal(entries, False)
    counts = bootstrap.resample_counts(layout.n_classes, n_boot=7)
    for split in ("stimulations", "neurons"):
        boot = readings.score_split(target, {"a": pred}, entries, split, layout, counts)["a"]["boot"]
        r_, c_ = np.nonzero(entries)
        unit = layout.class_index[r_] if split == "neurons" else layout.class_index[c_]
        for b in range(7):
            w = counts[b][unit]
            y, a = target[r_, c_], pred[r_, c_]
            my, ma = np.average(y, weights=w), np.average(a, weights=w)
            expect = np.average((y - my) * (a - ma), weights=w) / math.sqrt(
                np.average((y - my) ** 2, weights=w) * np.average((a - ma) ** 2, weights=w))
            assert boot[b] == pytest.approx(expect)


def test_bootstrap_resamples_classes_not_entries():
    """Two entries of one class always move together: equal per-class duplicates leave r unchanged
    when every entry of a class has the same values."""
    unit = np.repeat(np.arange(10), 30)
    rng = np.random.default_rng(2)
    y = np.repeat(rng.normal(size=10), 30)
    a = np.repeat(rng.normal(size=10), 30)
    counts = bootstrap.resample_counts(10, n_boot=50, seed=4)
    got = bootstrap.resampled_r(y, a, unit, counts)
    per_class = bootstrap.resampled_r(y[::30], a[::30], np.arange(10), counts)
    assert np.allclose(got, per_class, equal_nan=True)


def scores_of(**rs):
    boot = np.zeros(6)
    return {k: {"readable": v is not None, "r": v, "boot": boot + (v or 0)} for k, v in rs.items()}


def test_m_has_its_floor_and_takes_the_best_reference():
    s = scores_of(masked=0.30, **{"floor-C": 0.10, "H0-cal": 0.12, "C.2-cal": None})
    m = readings.minimum_effect(s)
    assert m["ref_arm"] == "H0-cal" and m["r_ref"] == 0.12
    assert m["m"] == pytest.approx(0.25 * 0.18)
    s["masked"]["r"] = 0.13
    assert readings.minimum_effect(s)["m"] == C.M0
    s["masked"]["r"] = 0.05  # below every reference: still the floor, never negative
    assert readings.minimum_effect(s)["m"] == C.M0


def test_rank_ties_count_against_masked():
    boot = np.zeros(5)
    sc = {"masked": {"readable": True, "r": 0.2, "boot": boot + .2}}
    for k in range(19):
        sc[f"d{k}"] = {"readable": True, "r": 0.2 if k == 0 else 0.1, "boot": boot + .1}
    f = readings.family(sc, "masked", [f"d{k}" for k in range(19)])
    assert f["rank"] == 2 and f["n_ranked"] == 20
    assert not readings.family_passes(f, 0.0)


def test_the_family_median_is_recomputed_in_every_resample():
    rng = np.random.default_rng(5)
    n_boot = 400
    sc = {"masked": {"readable": True, "r": 0.3, "boot": 0.3 + 0.01 * rng.normal(size=n_boot)}}
    draws = []
    for k in range(5):
        sc[f"d{k}"] = {"readable": True, "r": 0.1, "boot": 0.1 + 0.2 * rng.normal(size=n_boot)}
        draws.append(f"d{k}")
    f = readings.family(sc, "masked", draws)
    per_resample = np.median(np.stack([sc[d]["boot"] for d in draws]), axis=0)
    assert f["ci"] == pytest.approx(bootstrap.interval(sc["masked"]["boot"] - per_resample))
    fixed = bootstrap.interval(sc["masked"]["boot"] - 0.1)
    assert abs(f["ci"][0] - fixed[0]) > 0.05  # a fixed median would give a far narrower interval


def test_contrasts_are_paired():
    rng = np.random.default_rng(6)
    shared = rng.normal(size=500)
    a = {"readable": True, "r": 0.3, "boot": 0.3 + shared}
    b = {"readable": True, "r": 0.2, "boot": 0.2 + shared + 0.01 * rng.normal(size=500)}
    c = readings.contrast(a, b)
    assert c["ci"] == pytest.approx(bootstrap.interval(a["boot"] - b["boot"]))
    assert c["ci"][1] - c["ci"][0] < 0.1  # paired: the shared resampling noise cancels
    assert readings.contrast(a, 0.25)["ci"] == pytest.approx(bootstrap.interval(a["boot"] - 0.25))


def test_a_readable_arm_must_cover_every_entry():
    """Amendment 1, item 5: each arm's defined-entry count equals the set's size."""
    layout = splits.make_layout(CLASSES)
    target = np.ones((K, K))
    entries = np.zeros((K, K), bool)
    entries[0, 1:10] = True
    pred = np.arange(K * K, dtype=float).reshape(K, K)
    counts = bootstrap.resample_counts(layout.n_classes, n_boot=3)
    readings.score_split(target + pred, {"a": pred}, entries, "stimulations", layout, counts)
    pred[0, 5] = np.nan
    with pytest.raises(AssertionError, match="defined on 8 of the 9"):
        readings.score_split(target + pred, {"a": pred}, entries, "stimulations", layout, counts)


# ---- item 2: caps -------------------------------------------------------------------------------------


def test_the_cap_is_part_of_every_task_identity(tmp_path):
    _, _, T, _, _ = poisoned_world()
    d = run_dir_with(tmp_path, T, {name: supports.sparse_er(300, 170, 9900 + k)
                                   for k, name in enumerate(P.SELECTION_SET)})
    seen = []

    def stub(rd, task):
        seen.append(task)
        out = {"id": task["id"], "summary": {"readable": True}}
        if task["inner"] is not None:
            out["val_loss"] = 1.0
        else:
            out["pred"] = np.zeros(1)
        return out

    units = {("worms", None): ["masked", P.TIED, P.DENSE]}
    for long_cap in (C.MAX_ITER, C.CONVERGENCE_LONG):
        caps = P.caps(long_cap)
        choice = pipeline.select_lambdas(d, "c", "real", units, stub, workers=0, check_busy=False, caps=caps)
        pipeline.final_fits(d, "c", "real", units, choice, stub, workers=0, check_busy=False, caps=caps)
    by_cap = {}
    for t in seen:
        assert f"/it{t['max_iter']}/" in t["id"]
        by_cap.setdefault((t["arm"], t["max_iter"]), 0)
        by_cap[(t["arm"], t["max_iter"])] += 1
    # the long cap reaches sparse-support and tied fits, not dense; nothing was reused across caps
    assert ("masked", C.CONVERGENCE_LONG) in by_cap and (P.TIED, C.CONVERGENCE_LONG) in by_cap
    assert (P.DENSE, C.CONVERGENCE_LONG) not in by_cap
    assert by_cap[("masked", C.MAX_ITER)] == by_cap[("masked", C.CONVERGENCE_LONG)]


@pytest.mark.parametrize("short, long, triggered", [(1.0102, 1.0, True), (1.0099, 1.0, False)])
def test_the_convergence_rule(monkeypatch, tmp_path, short, long, triggered):
    """Amendment 1, item 2: the long cap applies when the visible loss at 500 is more than 1%
    (relative) above the loss at 2,000."""
    _, truth_s, T, _, _ = poisoned_world()
    d = run_dir_with(tmp_path, T, {"masked": truth_s})
    layout = splits.make_layout(CLASSES)
    calls = []

    def fake_fit(build, T_, mask, lam, next_lam, max_iter=C.MAX_ITER, constants=None):
        calls.append((lam, max_iter))
        return fitting.FitResult(lam, lam, True, False, True, max_iter,
                                 short if max_iter == C.CONVERGENCE_SHORT else long,
                                 short if max_iter == C.CONVERGENCE_SHORT else long, 0.0, 1.0, 1, 0.0)

    monkeypatch.setattr(stages.fitting, "fit", fake_fit)
    out = stages.convergence_check(d, T, layout)
    assert out["triggered"] == triggered
    assert calls == [(C.CONVERGENCE_LAMBDA, C.CONVERGENCE_SHORT), (C.CONVERGENCE_LAMBDA, C.CONVERGENCE_LONG)]
    assert C.CONVERGENCE_SHORT == 500 and C.CONVERGENCE_LONG == 2000 and C.CONVERGENCE_LAMBDA == 1e-2


def test_cap_hits_are_reported(tmp_path):
    ck = tmp_path / "checkpoints" / "truth"
    ck.mkdir(parents=True)
    rows = [{"id": "truth/masked/worms/masked/1e-02/it500/final", "summary": {"at_cap": True}},
            {"id": "truth/masked/worms/masked/1e-03/it500/inner0", "summary": {"at_cap": False}},
            {"id": "truth/masked/worms/tied/1e-03/it500/inner0", "summary": {"at_cap": True}}]
    (ck / "masked.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    hits = pipeline.cap_hits(tmp_path)
    assert hits["truth/masked|masked"]["fits"] == 2 and hits["truth/masked|masked"]["at_cap"] == 1
    assert hits["truth/masked|tied"]["at_cap"] == 1


# ---- item 6: unreadable, not refused ------------------------------------------------------------------


def test_an_arm_with_no_usable_lambda_pools_as_unreadable(tmp_path):
    _, truth_s, T, _, _ = poisoned_world()
    d = run_dir_with(tmp_path, T, {"masked": truth_s})
    layout = splits.make_layout(CLASSES)
    units = {("worms", None): ["masked"]}
    choice = {"worms|sparse": {"lambda": None, "extension": None}}
    ck, tasks = pipeline.final_fits(d, "u", "real", units, choice, lambda *a: pytest.fail("ran"),
                                    workers=0, check_busy=False)
    assert pipeline.pool(d, ck, tasks, T, layout)["worms"]["masked"] is None
    (row,) = pipeline.fit_summaries(ck, tasks)
    assert row["readable"] is False and "no lambda" in row["skipped"]


def test_a_split_with_an_unreadable_masked_is_scored_not_refused(tmp_path):
    from ceiling import scorer

    layout = splits.make_layout(CLASSES)
    rng = np.random.default_rng(8)
    T = rng.normal(size=(K, K))
    np.fill_diagonal(T, np.nan)
    s_fin = rng.random((K, K)) > .5
    np.fill_diagonal(s_fin, False)
    sets = splits.entry_sets(np.isfinite(T), s_fin, layout)
    np.savez(tmp_path / "entry_sets.npz", **{f"{s}/{f}": m for s, ms in sets.items() for f, m in enumerate(ms)})
    refs, _ = stages.reference_pool(T, layout, P.units(C.SPLITS), {"H0": rng.normal(size=(K, K)),
                                                                   "C.2": rng.normal(size=(K, K))})
    pooled = {s: {**refs[s], "masked": None} for s in C.SPLITS}
    oracle = {"gates": {"K": {"stimulations": True, "neurons": True},
                        "P": {"stimulations": True, "neurons": True, "worms": True},
                        "N": {"stimulations": False, "neurons": False}}}
    scorer.check_scoring_inputs(pooled, oracle, tmp_path)  # no refusal
    entry = np.load(tmp_path / "entry_sets.npz")
    out = report.readings_on_target(T, T, layout, pooled, stages.FAMILY_NAMES,
                                    {**oracle, "r_att": {}, "description_only": []}, entry, is_real=True)
    assert out["branches"]["stimulations"]["R-B"] == readings.UNREADABLE
    assert out["splits"]["stimulations"]["R-0"]["C.2 - H0"] is not None  # R-0 still reported
    pooled["stimulations"]["floor-C"] = pooled["stimulations"]["floor-C"].copy()
    pooled["stimulations"]["floor-C"][np.nonzero(sets["stimulations"][0])[0][0],
                                      np.nonzero(sets["stimulations"][0])[1][0]] = np.nan
    with pytest.raises(scorer.Refusal, match="undefined on 1"):
        scorer.check_scoring_inputs(pooled, oracle, tmp_path)


def test_real_stage_builds_references_on_every_split_when_k_fails(tmp_path, monkeypatch):
    """With K failing on held-out stimulations the learned arms run on held-out worms only, and
    the references are still built on every split for R-0."""
    rng, truth_s, T, _, _ = poisoned_world()
    d = run_dir_with(tmp_path, T, {"masked": truth_s})
    np.savez(d / "mechanistic.npz", H0=rng.normal(size=(K, K)), C2=rng.normal(size=(K, K)))
    (d / "preflight.json").write_text(json.dumps({"cuts": [], "caps": P.caps(C.MAX_ITER)}))
    (d / "oracles.json").write_text(json.dumps({"gates": {"K": {"stimulations": False, "neurons": True}}}))
    ran = []

    def fake_run_dataset(run_dir, prefix, dataset, arms_by_unit, layout, **kw):
        ran.extend(arms_by_unit)
        return {"choice": {}, "fits": [], "pooled": {"worms": {a: np.zeros((K, K)) for a in arms_by_unit[("worms", None)]}}}

    monkeypatch.setattr(stages.pipeline, "run_dataset", fake_run_dataset)
    monkeypatch.setattr(C, "BUSY_PATTERN", "no-such-process-9771x")
    out = stages.real_stage(d, workers=0)
    assert ran == [("worms", None)]
    pooled = stages.load_pooled(d)
    for split in C.SPLITS:
        assert pooled[split]["floor-C"] is not None and pooled[split]["H0"] is not None
    assert pooled["stimulations"]["masked"] is None and "stimulations|masked" in out["not_run"]
