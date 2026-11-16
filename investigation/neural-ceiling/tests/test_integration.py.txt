"""The scoring path end to end on synthetic data: oracle evaluation, the guarded registered reading
on a temporary repository, the branches, and the results files."""

import csv
import json

import numpy as np
import pytest

from ceiling import config as C
from ceiling import inputs, splits, stages
from ceiling import plan as P
from test_guard import REL, make_scorer, repo  # noqa: F401 - the fixture

K = C.N_MEASURED


def synthetic_world(seed=0):
    rng = np.random.default_rng(seed)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(K)])
    truth = rng.normal(size=(K, K))
    T = truth + rng.normal(size=(K, K))
    T[rng.random((K, K)) < 0.08] = np.nan
    np.fill_diagonal(T, np.nan)
    target = truth + 1.5 * rng.normal(size=(K, K))
    target[rng.random((K, K)) < 0.5] = np.nan
    np.fill_diagonal(target, np.nan)
    return rng, layout, truth, T, target


def pooled_for(rng, truth, T, layout, splits_present, arms, quality):
    pooled = {}
    for split in splits_present:
        pooled[split] = {a: truth + quality(a) * rng.normal(size=(K, K)) for a in arms(split)}
    refs, _ = stages.reference_pool(T, layout, P.units(splits_present),
                                    {"H0": rng.normal(size=(K, K)), "C.2": rng.normal(size=(K, K))})
    for split in splits_present:
        pooled[split].update(refs[split])
    return pooled


def entry_from(T, target, layout):
    sets = splits.entry_sets(np.isfinite(T), np.isfinite(target), layout)
    return {f"{s}/{f}": m for s, ms in sets.items() for f, m in enumerate(ms)}


def test_oracle_evaluation_and_gates(tmp_path):
    rng, layout, truth, T, target = synthetic_world(1)
    entry = entry_from(T, target, layout)
    quality = lambda a: 0.3 if a == "masked" else 3.0  # noqa: E731 - masked far better
    results, targets = {}, {}
    pipelinefits = [{"split": s, "data_loss": 1e-6} for s in C.HELD_OUT_SPLITS for _ in range(5)]
    results["K"] = {"pooled": {s: {"masked": truth.copy()} for s in C.HELD_OUT_SPLITS}, "fits": pipelinefits}
    (tmp_path / "datasets").mkdir()
    np.savez(tmp_path / "datasets/K.npz", T=T)
    targets["K"] = np.where(np.isfinite(target), truth, np.nan)
    for seed in C.SEEDS_P:
        name = f"P{seed}"
        results[name] = {"pooled": {s: {a: truth + quality(a) * rng.normal(size=(K, K)) for a in P.p_arms(s)}
                                    for s in C.SPLITS}}
        np.savez(tmp_path / f"datasets/{name}.npz", T=T)
        targets[name] = target
    for seed in C.N_TRUTH_SEEDS:
        name = f"N{seed}"
        results[name] = {"pooled": {s: {a: truth + 3.0 * rng.normal(size=(K, K))
                                        for a in P.n_arms(f"nullD:{seed}")} for s in C.HELD_OUT_SPLITS}}
        np.savez(tmp_path / f"datasets/{name}.npz", T=T)
        targets[name] = target
    out = stages.evaluate_oracles(tmp_path, results, targets, layout, entry, C.HELD_OUT_SPLITS, ())
    gates = out["gates"]
    assert gates["K"] == {"stimulations": True, "neurons": True}
    assert gates["P"]["stimulations"] and gates["P"]["neurons"]
    assert gates["N"] == {"stimulations": False, "neurons": False}  # masked is no better there
    assert set(out["r_att"]) == set(C.HELD_OUT_SPLITS)
    assert out["detail"]["P"]["P9773"]["stimulations"]["fold_heterogeneity"][0]["n"] > 0
    json.dumps(out, default=float)  # serialisable


def test_registered_reading_on_a_temporary_repository(repo, tmp_path):  # noqa: F811
    s, stams, sha = make_scorer(repo, tmp_path)
    T = inputs.load_T(stams, sha)
    rng = np.random.default_rng(2)
    layout = splits.make_layout([f"c{k % 74:02d}" for k in range(K)])
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    s.entry_sets(T, layout, out_dir)
    truth = T  # the synthetic S is 0.5 T + noise, so arms built on T predict it
    quality = lambda a: 0.5 if a == "masked" else 4.0  # noqa: E731
    pooled = pooled_for(rng, truth, T, layout, C.SPLITS, lambda sp: P.real_arms(sp), quality)
    pooled["neurons"]["nullR:9781"] = None  # one unreadable draw
    oracle = {"gates": {"K": {"stimulations": True, "neurons": True},
                        "P": {"stimulations": True, "neurons": True, "worms": True},
                        "N": {"stimulations": False, "neurons": False}},
              "r_att": {sp: {"r_att": 0.3} for sp in C.HELD_OUT_SPLITS}, "description_only": []}
    results = s.registered_scores(T, layout, pooled, stages.FAMILY_NAMES, oracle, out_dir)
    b = results["branches"]
    assert b["stimulations"]["R-B"] == "Anatomy helps beyond degree and reciprocity statistics."
    assert b["stimulations"]["R-A"]["masked"] == "above the reference"
    assert b["worms"]["R-C"] == "Anatomy helps at equal size, beyond random sparsity."
    nr = results["splits"]["neurons"]["R-B"]["null-R"]
    assert nr["readable_draws"] == 18 and nr["n_ranked"] == 19
    assert results["splits"]["stimulations"]["R-0"]["C.2 - H0"] is not None
    with pytest.raises(Exception, match="already read"):
        s.registered_scores(T, layout, pooled, stages.FAMILY_NAMES, oracle, out_dir)
    data = json.loads((out_dir / "results.json").read_text())
    assert data["branches"] == json.loads(json.dumps(b))
    raw = (out_dir / "results_per_arm.csv").read_bytes()
    assert b"\r\n" not in raw
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert {r["split"] for r in rows} == set(C.SPLITS)
    masked = [r for r in rows if r["arm"] == "masked" and r["split"] == "stimulations"][0]
    assert masked["family"] == "masked" and masked["readable"] == "1" and masked["fold_4"] != ""
    assert [r for r in rows if r["arm"] == "nullR:9781" and r["split"] == "neurons"][0]["readable"] == "0"
    log = (repo / REL.parent / "READS.md").read_text()
    assert log.count("- quantity: `") == 2
