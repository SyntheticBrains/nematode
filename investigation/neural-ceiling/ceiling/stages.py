"""The run in its registered order: setup, preflight, the oracles, the real run, the scoring.

Each stage is resumable: fits are checkpointed, and each stage's outputs are files in the run
directory. Only the scorer reads S; preflight and the oracles get its pattern and section 3's
summaries from the files the scorer wrote.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ceiling import config as C
from ceiling import fitting, floors, inputs, oracles, pipeline, report, splits, supports
from ceiling import plan as P
from ceiling.bootstrap import resample_counts
from ceiling.readings import contrast, family, minimum_effect, score_split
from ceiling.runner import refuse_if_busy
from ceiling.scorer import Scorer

FAMILY_NAMES = {"null-D": P.NULL_D, "null-R": P.NULL_R, "sparse-ER": P.ER}


def make_scorer(confirm: bool, amendment: str | None = None) -> Scorer:
    """The guarded scorer on the real registration and S (tests replace this seam)."""
    return Scorer(confirm, amendment)


def _json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=float))


def _load(path: Path):
    return json.loads(Path(path).read_text())


def layout_of(run_dir: Path):
    return splits.make_layout(_load(Path(run_dir) / "layout.json")["classes"])


# ---- setup ----------------------------------------------------------------------------------------


def setup(run_dir: Path) -> dict:
    """Inputs, layout, every support and the zero-shot responses. Reads T, never S."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    names = inputs.load_names()
    T = inputs.load_T()
    names302 = inputs.connectome_names()
    measured = inputs.measured_index(names, names302)
    if not (run_dir / "layout.json").exists():
        _json(run_dir / "layout.json", {"names": names, "classes": splits.measured_classes(names)})
    if not (run_dir / "supports.npz").exists():
        built = {"masked": supports.anatomical(names302)}
        wild = supports.brain_support(None)
        if wild.keys() != built["masked"].keys():
            raise AssertionError("the anatomical union differs from the wild-type brain's support")
        for seed in C.SEEDS_NULL_D:
            built[f"nullD:{seed}"] = supports.null_d(seed)
        for seed in C.SEEDS_NULL_R:
            built[f"nullR:{seed}"] = supports.null_r(built["masked"], seed)
        for seed in C.SEEDS_ER:
            built[f"er:{seed}"] = supports.sparse_er(built["masked"].size, C.N_CONNECTOME, seed)
        pipeline.save_supports(run_dir, built, measured)
        _json(run_dir / "supports.json", {
            name: {"size": s.size, "reciprocal_entries": s.reciprocal_entries(), **s.report}
            for name, s in built.items()})
    if not (run_dir / "mechanistic.npz").exists():
        from ceiling import mechanistic

        resp = mechanistic.responses(names)
        np.savez_compressed(run_dir / "mechanistic.npz", H0=resp["H0"], C2=resp["C.2"])
        _json(run_dir / "mechanistic.json", {"parity_max_abs": resp["parity"]})
    pipeline.save_dataset(run_dir, "real", T)
    return {"T_finite": int(np.isfinite(T).sum()), "measured": len(measured)}


def mechanistic_of(run_dir: Path) -> dict[str, np.ndarray]:
    m = np.load(Path(run_dir) / "mechanistic.npz")
    return {"H0": m["H0"], "C.2": m["C2"]}


# ---- references ------------------------------------------------------------------------------------


def reference_pool(T: np.ndarray, layout, units, mechanistic: dict | None) -> tuple[dict, dict]:
    """floor-ME, floor-T, floor-C (and H0, H0-cal, C.2, C.2-cal) pooled like the fitted arms."""
    pooled: dict[str, dict] = {}
    coefs: dict[str, dict] = {}
    for split, fold in units:
        masks = pipeline.unit_masks(T, layout, split, fold)
        refs = floors.references(T, masks["visible"], split, mechanistic)
        for name, ref in refs.items():
            target = pooled.setdefault(split, {}).setdefault(
                name, np.full((C.N_MEASURED, C.N_MEASURED), np.nan))
            target[masks["hidden"]] = ref["pred"][masks["hidden"]]
            coefs.setdefault(split, {})[f"{name}/{fold}"] = {"coef": ref["coef"], "n_fit": ref["n_fit"]}
    return pooled, coefs


# ---- preflight --------------------------------------------------------------------------------------


def preflight(run_dir: Path, confirm: bool, workers: int = C.WORKERS) -> dict:
    """Entry sets (via the scorer), C.2 - H0's interval width against T, one timed fit of each
    kind, the cost projection and the cuts. Fits real T; reads no S value."""
    refuse_if_busy()
    run_dir = Path(run_dir)
    T = np.load(run_dir / "datasets/real.npz")["T"]
    layout = layout_of(run_dir)
    if not (run_dir / "entry_sets.npz").exists():
        make_scorer(confirm).entry_sets(T, layout, run_dir)
    entry = np.load(run_dir / "entry_sets.npz")
    mech = mechanistic_of(run_dir)
    counts = resample_counts(layout.n_classes)
    widths = {}
    for split in C.SPLITS:
        folds = report.fold_entry_masks(entry, split)
        entries = np.any(np.stack(folds), axis=0) & np.isfinite(T)
        scores = score_split(T, {"H0": mech["H0"], "C.2": mech["C.2"]}, entries, split, layout, counts)
        c = contrast(scores["C.2"], scores["H0"])
        widths[split] = {"diff": c["diff"], "ci": c["ci"], "width": c["ci"][1] - c["ci"][0],
                         "entries": int(entries.sum())}
    timings = timed_fits(run_dir, T, layout)
    seconds = {k: v["seconds"] for k, v in timings.items()}
    cuts = P.cut_order(seconds)
    out = {"entry_sets": _load(run_dir / "entry_sets.json")["report"],
           "c2_minus_h0_against_T": widths, "timings": timings, "projection": cuts,
           "cuts": cuts["cuts"], "cut_order_registered": list(C.CUTS)}
    _json(run_dir / "preflight.json", out)
    return out


def timed_fits(run_dir: Path, T: np.ndarray, layout) -> dict:
    """One fit of each kind on held-out-stimulation fold 0, at the preflight lambda."""
    ctx = pipeline._context(str(run_dir))
    masks = pipeline.unit_masks(T, layout, "stimulations", 0)
    kinds = {"sparse": "masked", "sparse_nullD": "nullD:9741", "prior": "prior:masked",
             P.TIED: P.TIED, P.DENSE: P.DENSE, P.UNTIED: P.UNTIED,
             **{a: a for a in P.SWEEP}, "K": "masked"}
    out = {}
    for kind, arm in kinds.items():
        max_iter = C.MAX_ITER_K if kind == "K" else C.MAX_ITER
        lam = C.K_LAMBDA if kind == "K" else C.PREFLIGHT_LAMBDA
        start = time.perf_counter()
        r = fitting.fit(pipeline.builder(arm, ctx), T, masks["visible"], lam,
                        fitting.next_lambda(lam, list(C.LAMBDA_GRID)), max_iter)
        out[kind] = {"arm": arm, "seconds": time.perf_counter() - start, "n_iter": r.n_iter,
                     "at_cap": r.at_cap, "readable": r.readable, "moved": r.moved,
                     "n_params": r.n_params}
    tied = out[P.TIED]
    tied["both_move"] = bool(len(tied["moved"]) == 2 and all(m > 0 for m in tied["moved"]))
    return out


# ---- oracles ----------------------------------------------------------------------------------------


def _truth(run_dir: Path, arm: str, workers: int) -> np.ndarray:
    """An arm fitted to all of T at the lambda of its own inner CV on T (random entries)."""
    unit = {("worms", None): [arm]}
    prefix = f"truth/{arm}"
    choice = pipeline.select_lambdas(run_dir, prefix, "real", unit, workers=workers,
                                     selection_override={"sparse": [arm]})
    ck, tasks = pipeline.final_fits(run_dir, prefix, "real", unit, choice, workers=workers)
    (tid,) = tasks
    if not ck.done[tid]["summary"]["readable"]:
        raise RuntimeError(f"the truth {arm} is unreadable")
    M = np.full((C.N_MEASURED, C.N_MEASURED), np.nan)
    M[splits.off_diagonal()] = ck.prediction(tid)
    return M


def oracles_stage(run_dir: Path, confirm: bool, workers: int = C.WORKERS) -> dict:
    refuse_if_busy()
    run_dir = Path(run_dir)
    pre = _load(run_dir / "preflight.json")
    cuts = tuple(pre["cuts"])
    if pre["projection"]["action"] != "proceed":
        raise RuntimeError("preflight projects over budget after every cut: stop and amend")
    T = np.load(run_dir / "datasets/real.npz")["T"]
    layout = layout_of(run_dir)
    if not (run_dir / "section3.json").exists():
        make_scorer(confirm).section3(T, layout, run_dir)
    sec3 = _load(run_dir / "section3.json")
    cal = oracles.calibration(sec3)
    entry = np.load(run_dir / "entry_sets.npz")
    t_pat, s_pat = np.isfinite(T), np.asarray(entry["s_finite"])
    plan = P.datasets(cuts)
    held = P.oracle_splits(cuts)

    m_star = _truth(run_dir, "masked", workers)
    targets: dict[str, np.ndarray] = {}
    reports: dict[str, dict] = {}
    t_syn, s_syn = oracles.known_answer(m_star, t_pat, s_pat)
    pipeline.save_dataset(run_dir, "K", t_syn)
    targets["K"] = s_syn
    for seed in C.SEEDS_P:
        t_syn, s_syn, rep = oracles.synthetic_pair(m_star, T - m_star, t_pat, s_pat, cal["rel_T"],
                                                   cal["rel_S"], np.random.default_rng(seed))
        pipeline.save_dataset(run_dir, f"P{seed}", t_syn)
        targets[f"P{seed}"], reports[f"P{seed}"] = s_syn, rep
    for truth_seed, noise_seed in zip(P.n_truths(cuts), C.SEEDS_N_NOISE):
        m_d = _truth(run_dir, f"nullD:{truth_seed}", workers)
        t_syn, s_syn, rep = oracles.synthetic_pair(m_d, T - m_d, t_pat, s_pat, cal["rel_T"],
                                                   cal["rel_S"], np.random.default_rng(noise_seed))
        pipeline.save_dataset(run_dir, f"N{truth_seed}", t_syn)
        targets[f"N{truth_seed}"], reports[f"N{truth_seed}"] = s_syn, rep
    np.savez_compressed(run_dir / "oracle_targets.npz", **targets)

    results: dict[str, dict] = {}
    ck, tasks = pipeline.final_fits(run_dir, "K", "K", plan["K"], {}, workers=workers,
                                    max_iter=C.MAX_ITER_K, fixed_lambda=C.K_LAMBDA)
    k_pool = pipeline.pool(run_dir, ck, tasks, T, layout)
    results["K"] = {"pooled": k_pool, "fits": pipeline.fit_summaries(ck, tasks)}
    for name in plan:
        if name in ("real", "K"):
            continue
        results[name] = pipeline.run_dataset(run_dir, name, name, plan[name], layout, workers=workers)
    out = evaluate_oracles(run_dir, results, targets, layout, entry, held, cuts)
    out.update({"calibration": cal, "noise": reports, "cuts": list(cuts)})
    _json(run_dir / "oracles.json", out)
    return out


def _with_references(pooled: dict, t_syn: np.ndarray, layout, splits_present) -> dict:
    units = [u for u in P.units(splits_present)]
    refs, _ = reference_pool(t_syn, layout, units, None)
    merged = {s: dict(pooled.get(s, {})) for s in splits_present}
    for s in splits_present:
        merged[s].update(refs.get(s, {}))
    return merged


def evaluate_oracles(run_dir: Path, results: dict, targets: dict, layout, entry, held, cuts) -> dict:
    counts = resample_counts(layout.n_classes)
    gates: dict = {"K": {}, "P": {}, "N": {}}
    detail: dict = {"K": {}, "P": {}, "N": {}}
    # K
    for split in held:
        sc = report.split_scores(targets["K"], results["K"]["pooled"][split], entry, split, layout, counts)
        r = sc["masked"]["r"] if sc["masked"].get("readable") else float("nan")
        losses = [f["data_loss"] for f in results["K"]["fits"] if f["split"] == split]
        detail["K"][split] = oracles.k_gate(r, losses)
        gates["K"][split] = detail["K"][split]["passes"]
    # P
    draws = {split: [] for split in held}
    worm_lines, r_masked = [], {split: [] for split in held}
    for seed in C.SEEDS_P:
        name = f"P{seed}"
        t_syn = np.load(Path(run_dir) / f"datasets/{name}.npz")["T"]
        pooled = _with_references(results[name]["pooled"], t_syn, layout, held + ("worms",))
        per = {}
        for split in held + ("worms",):
            preds = dict(pooled[split])
            if split == "worms":
                preds["T"] = t_syn
            sc = report.split_scores(targets[name], preds, entry, split, layout, counts)
            rd = report.split_readings(sc, split, FAMILY_NAMES, ("floor-C",))
            per[split] = {"readings": rd, "scores": report._strip(sc),
                          "fold_heterogeneity": fold_heterogeneity(targets[name], pooled[split].get("masked"),
                                                                   entry, split)}
            if split == "worms":
                worm_lines.append(rd["R-C"]["denoising"])
            else:
                draws[split].append({"m": rd["m"], **rd["R-B"]})
                r_masked[split].append(sc["masked"]["r"] if sc["masked"].get("readable") else float("nan"))
        per["synthetic_ceiling"] = oracles.synthetic_ceiling(t_syn, targets[name])
        detail["P"][name] = per
    for split in held:
        g = oracles.p_split_gate(draws[split])
        gates["P"][split] = g["passes"]
        detail["P"][f"gate/{split}"] = g
    g = oracles.p_worm_gate(worm_lines)
    gates["P"]["worms"] = g["passes"]
    detail["P"]["gate/worms"] = g
    # N
    truths = {split: [] for split in held}
    for seed in P.n_truths(cuts):
        name = f"N{seed}"
        t_syn = np.load(Path(run_dir) / f"datasets/{name}.npz")["T"]
        pooled = _with_references(results[name]["pooled"], t_syn, layout, held)
        for split in held:
            sc = report.split_scores(targets[name], pooled[split], entry, split, layout, counts)
            others = [d for d in P.NULL_D if d != f"nullD:{seed}" and d in sc]
            m = minimum_effect(sc, ("floor-C",))["m"]
            truths[split].append({"m": m, "null-D": family(sc, "masked", others)})
    for split in held:
        g = oracles.n_gate(truths[split])
        gates["N"][split] = g["fails"]
        detail["N"][split] = {"gate": g, "truths": truths[split]}
    r_att = {split: oracles.r_att(r_masked[split]) for split in held}
    description_only = [s for s in C.HELD_OUT_SPLITS if s not in held]
    for s in description_only:
        gates["K"][s] = gates["P"][s] = False
        gates["N"][s] = False
    return {"gates": gates, "detail": detail, "r_att": r_att, "description_only": description_only}


def fold_heterogeneity(target, pred, entry, split) -> list[dict] | None:
    """Per fold, the least-squares scale and offset of the target on the prediction."""
    if pred is None or split == "worms":
        return None
    out = []
    for mask in report.fold_entry_masks(entry, split):
        y, x = target[mask], pred[mask]
        keep = np.isfinite(y) & np.isfinite(x)
        if keep.sum() < 3:
            out.append({"scale": float("nan"), "offset": float("nan"), "n": int(keep.sum())})
            continue
        scale, offset = np.polyfit(x[keep], y[keep], 1)
        out.append({"scale": float(scale), "offset": float(offset), "n": int(keep.sum())})
    return out


# ---- the real run and its scoring ---------------------------------------------------------------------


def real_stage(run_dir: Path, workers: int = C.WORKERS) -> dict:
    refuse_if_busy()
    run_dir = Path(run_dir)
    pre, orc = _load(run_dir / "preflight.json"), _load(run_dir / "oracles.json")
    cuts = tuple(pre["cuts"])
    T = np.load(run_dir / "datasets/real.npz")["T"]
    layout = layout_of(run_dir)
    arms = P.datasets(cuts)["real"]
    if not orc["gates"]["K"]["stimulations"]:
        arms = {u: a for u, a in arms.items() if u[0] == "worms"}  # stop rule: only held-out worms
    result = pipeline.run_dataset(run_dir, "real", "real", arms, layout, workers=workers)
    present = tuple(sorted({u[0] for u in arms}))
    refs, coefs = reference_pool(T, layout, P.units(present), mechanistic_of(run_dir))
    pooled = {s: {**result["pooled"].get(s, {}), **refs.get(s, {})} for s in present}
    arrays = {f"{s}|{a}": m for s, d in pooled.items() for a, m in d.items() if m is not None}
    unreadable = [f"{s}|{a}" for s, d in pooled.items() for a, m in d.items() if m is None]
    np.savez_compressed(run_dir / "pooled_real.npz", **arrays)
    _json(run_dir / "real_fits.json", {"choice": result["choice"], "fits": result["fits"],
                                       "calibration_coefficients": coefs, "unreadable": unreadable,
                                       "splits": list(present)})
    return {"splits": list(present), "unreadable": unreadable}


def load_pooled(run_dir: Path) -> dict:
    meta = _load(Path(run_dir) / "real_fits.json")
    archive = np.load(Path(run_dir) / "pooled_real.npz")
    pooled: dict[str, dict] = {s: {} for s in meta["splits"]}
    for key in archive.files:
        s, a = key.split("|", 1)
        pooled[s][a] = archive[key]
    for key in meta["unreadable"]:
        s, a = key.split("|", 1)
        pooled[s][a] = None
    return pooled


def score_stage(run_dir: Path, confirm: bool, amendment: str | None = None) -> dict:
    run_dir = Path(run_dir)
    T = np.load(run_dir / "datasets/real.npz")["T"]
    layout = layout_of(run_dir)
    orc = _load(run_dir / "oracles.json")
    return make_scorer(confirm, amendment).registered_scores(T, layout, load_pooled(run_dir),
                                                             FAMILY_NAMES, orc, run_dir)
