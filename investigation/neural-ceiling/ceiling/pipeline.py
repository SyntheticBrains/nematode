"""One data set through section 6: lambda selection by inner CV, the one-time edge extension, the
final fits, and the held-out predictions pooled over folds. Plus the worker's `execute`.

A run directory holds the context every worker reads: layout.json (the 154 names and classes),
supports.npz (every support, with counts) and datasets/<name>.npz (the training matrix of each data
set). Workers never see S.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from ceiling import config as C
from ceiling import fitting, models, splits
from ceiling import plan as P
from ceiling.runner import run_tasks
from ceiling.supports import Support

# ---- context --------------------------------------------------------------------------------------


def save_supports(run_dir: Path, supports: dict[str, Support], measured: np.ndarray) -> None:
    arrays = {"measured": measured}
    for name, s in supports.items():
        for key in ("rows", "cols", "chem", "gap"):
            arrays[f"{name}/{key}"] = getattr(s, key)
        arrays[f"{name}/n"] = np.array(s.n)
    np.savez_compressed(Path(run_dir) / "supports.npz", **arrays)


def save_dataset(run_dir: Path, name: str, T: np.ndarray) -> None:
    path = Path(run_dir) / "datasets" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, T=T)


@functools.cache
def _context(run_dir: str) -> dict:
    meta = json.loads((Path(run_dir) / "layout.json").read_text())
    layout = splits.make_layout(meta["classes"])
    archive = np.load(Path(run_dir) / "supports.npz")
    supports = {}
    for key in archive.files:
        if key.endswith("/rows"):
            name = key[: -len("/rows")]
            supports[name] = Support(name, int(archive[f"{name}/n"]), archive[f"{name}/rows"],
                                     archive[f"{name}/cols"], archive[f"{name}/chem"],
                                     archive[f"{name}/gap"])
    tied_rank = models.matched_tied_rank(supports["masked"].size)
    return {"layout": layout, "supports": supports, "measured": archive["measured"],
            "tied_rank": tied_rank}


@functools.cache
def _dataset(run_dir: str, name: str) -> np.ndarray:
    return np.load(Path(run_dir) / "datasets" / f"{name}.npz")["T"]


def builder(arm: str, ctx: dict):
    """A function that builds the arm's model afresh (its initialisation included)."""
    fam = P.family_of(arm)
    k = C.N_MEASURED
    if fam in ("masked", "null-D", "null-R", "sparse-ER"):
        s = ctx["supports"][P.support_of(arm)]
        return lambda: models.SparseSeries(s.rows, s.cols, s.n, ctx["measured"])
    if fam == "masked-prior":
        s = ctx["supports"][P.support_of(arm)]
        return lambda: models.MaskedPrior(s.rows, s.cols, s.n, ctx["measured"], s.chem, s.gap)
    if arm == P.TIED:
        return lambda: models.Tied(k, ctx["tied_rank"])
    if fam == "tied-sweep":
        return lambda: models.Tied(k, int(arm.split(":")[1]))
    if fam == "dense":
        return lambda: models.Dense(k)
    if fam == "untied":
        return lambda: models.Untied(k, int(arm.split(":")[1]))
    raise ValueError(arm)


def unit_masks(T: np.ndarray, layout, split: str, fold: int | None) -> dict:
    t_finite = np.isfinite(T)
    in_fold = None if fold is None else layout.in_fold(fold)
    visible = splits.visible_mask(t_finite, split, in_fold)
    n = len(T)
    hidden = (splits.off_diagonal(n) if split == "worms"
              else splits.hidden_mask(split, in_fold, n) & splits.off_diagonal(n))
    return {"visible": visible, "hidden": hidden, "in_fold": in_fold,
            "unit_var": float(np.var(T[visible]))}


def execute(run_dir: str, task: dict) -> dict:
    """Fit one task; return its summary, its validation loss or its held-out prediction."""
    ctx = _context(run_dir)
    T = _dataset(run_dir, task["dataset"])
    layout = ctx["layout"]
    masks = unit_masks(T, layout, task["split"], task["fold"])
    if task["inner"] is not None:
        problems = splits.inner_problems(task["split"], np.isfinite(T), layout, task["fold"])
        train, validation = problems[task["inner"]]
    else:
        train, validation = masks["visible"], None
    result = fitting.fit(builder(task["arm"], ctx), T, train, task["lam"], task["next_lam"],
                         task["max_iter"])
    out = {"id": task["id"], "summary": result.summary()}
    if validation is not None:
        out["val_loss"] = fitting.validation_mse(result.pred, T, validation, masks["unit_var"])
    else:
        out["pred"] = None if result.pred is None else result.pred[masks["hidden"]]
    return out


# ---- stages ---------------------------------------------------------------------------------------


def _unit_key(unit) -> str:
    split, fold = unit
    return split if fold is None else f"{split}/{fold}"


def _task(prefix: str, dataset: str, unit, arm: str, lam: float, grid: list[float], inner,
          max_iter: int) -> dict:
    split, fold = unit
    tag = "final" if inner is None else f"inner{inner}"
    return {"id": f"{prefix}/{_unit_key(unit)}/{arm}/{lam:.0e}/{tag}", "dataset": dataset,
            "split": split, "fold": fold, "inner": inner, "arm": arm, "lam": lam,
            "next_lam": fitting.next_lambda(lam, grid), "max_iter": max_iter}


def _selection_tasks(prefix, dataset, unit, group, lams, grid, selection) -> list[dict]:
    return [_task(prefix, dataset, unit, arm, lam, grid, inner, C.MAX_ITER)
            for arm in selection for lam in lams for inner in range(C.N_INNER)]


def select_lambdas(run_dir: Path, prefix: str, dataset: str, arms_by_unit: dict, execute_fn=execute,
                   workers: int = C.WORKERS, selection_override: dict | None = None,
                   check_busy: bool = True) -> dict:
    """Inner-CV lambda per unit and group, with one edge extension; returns the record."""
    grid = list(C.LAMBDA_GRID)
    choice: dict[str, dict] = {}
    groups = {u: sorted({P.group_of(a) for a in arms}) for u, arms in arms_by_unit.items()}
    sel = {g: (selection_override or {}).get(g, P.selection_arms(g))
           for gs in groups.values() for g in gs}
    tasks = [t for u, gs in groups.items() for g in gs
             for t in _selection_tasks(prefix, dataset, u, g, grid, grid, sel[g])]
    ck = run_tasks(tasks, run_dir, prefix, execute_fn, workers, check_busy=check_busy)
    extensions = []
    for u, gs in groups.items():
        for g in gs:
            losses = _summed(ck, prefix, dataset, u, sel[g], grid, grid)
            lam = fitting.choose(losses)
            ext = None if lam is None else fitting.edge_extension(lam, grid)
            choice[f"{_unit_key(u)}|{g}"] = {"unit": list(u), "group": g, "losses": losses,
                                             "lambda": lam, "extension": ext}
            if ext is not None:
                extensions += _selection_tasks(prefix, dataset, u, g, [ext], sorted(grid + [ext]), sel[g])
    ck = run_tasks(extensions, run_dir, prefix, execute_fn, workers, check_busy=check_busy)
    for rec in choice.values():
        if rec["extension"] is None:
            continue
        u, g, ext = tuple(rec["unit"]), rec["group"], rec["extension"]
        full = sorted(grid + [ext])
        rec["losses"] = _summed(ck, prefix, dataset, u, sel[g], full, full)
        rec["lambda"] = fitting.choose(rec["losses"])
    return choice


def _summed(ck, prefix, dataset, unit, selection, lams, grid) -> dict[float, float]:
    out = {}
    for lam in lams:
        total = 0.0
        for arm in selection:
            for inner in range(C.N_INNER):
                tid = _task(prefix, dataset, unit, arm, lam, grid, inner, C.MAX_ITER)["id"]
                total += ck.done[tid]["val_loss"]
        out[lam] = total
    return out


def final_fits(run_dir: Path, prefix: str, dataset: str, arms_by_unit: dict, choice: dict,
               execute_fn=execute, workers: int = C.WORKERS, max_iter: int | None = None,
               fixed_lambda: float | None = None, check_busy: bool = True):
    max_iter = C.MAX_ITER if max_iter is None else max_iter
    tasks = []
    for u, arms in arms_by_unit.items():
        for arm in arms:
            if fixed_lambda is not None:
                lam, grid = fixed_lambda, list(C.LAMBDA_GRID)
            else:
                rec = choice[f"{_unit_key(u)}|{P.group_of(arm)}"]
                lam = rec["lambda"]
                grid = sorted(set(C.LAMBDA_GRID) | ({rec["extension"]} if rec["extension"] else set()))
            if lam is None:
                continue  # every lambda's inner fits unreadable: the arm is unreadable here
            tasks.append(_task(prefix, dataset, u, arm, lam, grid, None, max_iter))
    ck = run_tasks(tasks, run_dir, prefix, execute_fn, workers, check_busy=check_busy)
    return ck, {t["id"]: t for t in tasks}


def pool(run_dir: Path, ck, tasks: dict, T: np.ndarray, layout) -> dict[str, dict]:
    """Held-out predictions pooled over each split's folds: {split: {arm: matrix or None}}.
    An arm is readable on a split only when every fold's fit is."""
    if C.ARM_READABLE_RULE != "all_folds":
        raise NotImplementedError(C.ARM_READABLE_RULE)
    out: dict[str, dict] = {}
    by_split: dict[tuple[str, str], list[dict]] = {}
    for t in tasks.values():
        by_split.setdefault((t["split"], t["arm"]), []).append(t)
    for (split, arm), ts in by_split.items():
        expected = 1 if split == "worms" else C.N_FOLDS
        matrix = np.full((C.N_MEASURED, C.N_MEASURED), np.nan)
        ok = len(ts) == expected
        for t in ts:
            record = ck.done[t["id"]]
            pred = ck.prediction(t["id"])
            if not record["summary"]["readable"] or pred is None:
                ok = False
                continue
            matrix[unit_masks(T, layout, split, t["fold"])["hidden"]] = pred
        out.setdefault(split, {})[arm] = matrix if ok else None
    return out


def fit_summaries(ck, tasks: dict) -> list[dict]:
    rows = []
    for tid, t in tasks.items():
        s = ck.done[tid]["summary"]
        rows.append({"id": tid, "split": t["split"], "fold": t["fold"], "arm": t["arm"],
                     **{k: s[k] for k in ("lam_requested", "lam", "readable", "refit", "at_cap",
                                          "n_iter", "loss", "data_loss", "n_params", "seconds",
                                          "moved", "shares")}})
    return rows


def run_dataset(run_dir: Path, prefix: str, dataset: str, arms_by_unit: dict, layout,
                execute_fn=execute, workers: int = C.WORKERS, check_busy: bool = True) -> dict:
    """Selection, finals and pooling for one data set; resumable from its checkpoint."""
    choice = select_lambdas(run_dir, prefix, dataset, arms_by_unit, execute_fn, workers,
                            check_busy=check_busy)
    ck, tasks = final_fits(run_dir, prefix, dataset, arms_by_unit, choice, execute_fn, workers,
                           check_busy=check_busy)
    T = np.load(Path(run_dir) / "datasets" / f"{dataset}.npz")["T"]
    return {"choice": choice, "pooled": pool(run_dir, ck, tasks, T, layout),
            "fits": fit_summaries(ck, tasks)}

