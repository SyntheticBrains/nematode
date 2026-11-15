"""One data set through section 6: lambda selection by inner CV, the one-time edge extension, the
final fits, and the held-out predictions pooled over folds. Plus the worker's `execute`.

A run directory holds the context every worker reads: layout.json (the 154 names and classes),
supports.npz (every support, with counts) and datasets/<name>.npz (the training matrix of each data
set). Workers never see S.
"""

from __future__ import annotations

import functools
import json
import math
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


def save_dataset(run_dir: Path, name: str, T: np.ndarray,
                 constants: tuple[float, float] | None = None) -> None:
    """A data set's training matrix; an oracle's also carries its truth fit's (mean, SD), which
    every fit on it standardises with (amendment 1, item 1)."""
    path = Path(run_dir) / "datasets" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if constants is None else {"mu": np.float64(constants[0]), "sigma": np.float64(constants[1])}
    np.savez_compressed(path, T=T, **extra)


def load_dataset(run_dir: Path, name: str) -> tuple[np.ndarray, tuple[float, float] | None]:
    archive = np.load(Path(run_dir) / "datasets" / f"{name}.npz")
    constants = (float(archive["mu"]), float(archive["sigma"])) if "mu" in archive.files else None
    return archive["T"], constants


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
def _dataset(run_dir: str, name: str) -> tuple[np.ndarray, tuple[float, float] | None]:
    return load_dataset(Path(run_dir), name)


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
    T, constants = _dataset(run_dir, task["dataset"])
    layout = ctx["layout"]
    masks = unit_masks(T, layout, task["split"], task["fold"])
    if task["inner"] is not None:
        problems = splits.inner_problems(task["split"], np.isfinite(T), layout, task["fold"])
        train, validation = problems[task["inner"]]
    else:
        train, validation = masks["visible"], None
    result = fitting.fit(builder(task["arm"], ctx), T, train, task["lam"], task["next_lam"],
                         task["max_iter"], constants=constants)
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
    """A fit's identity includes its iteration cap, so no fit is reused across a change of cap."""
    split, fold = unit
    tag = "final" if inner is None else f"inner{inner}"
    return {"id": f"{prefix}/{_unit_key(unit)}/{arm}/{lam:.0e}/it{max_iter}/{tag}", "dataset": dataset,
            "split": split, "fold": fold, "inner": inner, "arm": arm, "lam": lam,
            "next_lam": fitting.next_lambda(lam, grid), "max_iter": max_iter}


def _selection_tasks(prefix, dataset, unit, group, lams, grid, selection, caps) -> list[dict]:
    return [_task(prefix, dataset, unit, arm, lam, grid, inner, P.cap_of(arm, caps))
            for arm in selection for lam in lams for inner in range(C.N_INNER)]


def select_lambdas(run_dir: Path, prefix: str, dataset: str, arms_by_unit: dict, execute_fn=execute,
                   workers: int = C.WORKERS, selection_override: dict | None = None,
                   check_busy: bool = True, caps: dict | None = None) -> dict:
    """Inner-CV lambda per unit and group, with one edge extension; returns the record."""
    grid = list(C.LAMBDA_GRID)
    choice: dict[str, dict] = {}
    groups = {u: sorted({P.group_of(a) for a in arms}) for u, arms in arms_by_unit.items()}
    sel = {g: (selection_override or {}).get(g, P.selection_arms(g))
           for gs in groups.values() for g in gs}
    tasks = [t for u, gs in groups.items() for g in gs
             for t in _selection_tasks(prefix, dataset, u, g, grid, grid, sel[g], caps)]
    ck = run_tasks(tasks, run_dir, prefix, execute_fn, workers, check_busy=check_busy)
    extensions = []
    for u, gs in groups.items():
        for g in gs:
            losses = _summed(ck, prefix, dataset, u, sel[g], grid, grid, caps)
            lam = fitting.choose(losses)
            ext = None if lam is None else fitting.edge_extension(lam, grid)
            choice[f"{_unit_key(u)}|{g}"] = {"unit": list(u), "group": g, "losses": losses,
                                             "lambda": lam, "extension": ext}
            if ext is not None:
                extensions += _selection_tasks(prefix, dataset, u, g, [ext], sorted(grid + [ext]),
                                               sel[g], caps)
    ck = run_tasks(extensions, run_dir, prefix, execute_fn, workers, check_busy=check_busy)
    for rec in choice.values():
        if rec["extension"] is None:
            continue
        u, g, ext = tuple(rec["unit"]), rec["group"], rec["extension"]
        full = sorted(grid + [ext])
        rec["losses"] = _summed(ck, prefix, dataset, u, sel[g], full, full, caps)
        rec["lambda"] = fitting.choose(rec["losses"])
    return choice


def _summed(ck, prefix, dataset, unit, selection, lams, grid, caps) -> dict[float, float]:
    out = {}
    for lam in lams:
        total = 0.0
        for arm in selection:
            for inner in range(C.N_INNER):
                tid = _task(prefix, dataset, unit, arm, lam, grid, inner, P.cap_of(arm, caps))["id"]
                total += ck.done[tid]["val_loss"]
        out[lam] = total
    return out


def final_fits(run_dir: Path, prefix: str, dataset: str, arms_by_unit: dict, choice: dict,
               execute_fn=execute, workers: int = C.WORKERS, max_iter: int | None = None,
               fixed_lambda: float | None = None, check_busy: bool = True, caps: dict | None = None):
    """The final fit of every arm at its chosen lambda. `max_iter`, if given, caps every arm (K);
    otherwise each arm's cap comes from `caps`. An arm with no usable lambda (every inner fit
    unreadable) is kept as a skipped task, so pooling reports it unreadable."""
    tasks = []
    for u, arms in arms_by_unit.items():
        for arm in arms:
            cap = max_iter if max_iter is not None else P.cap_of(arm, caps)
            if fixed_lambda is not None:
                lam, grid = fixed_lambda, list(C.LAMBDA_GRID)
            else:
                rec = choice[f"{_unit_key(u)}|{P.group_of(arm)}"]
                lam = rec["lambda"]
                grid = sorted(set(C.LAMBDA_GRID) | ({rec["extension"]} if rec["extension"] else set()))
            if lam is None:
                skipped = _task(prefix, dataset, u, arm, math.nan, list(C.LAMBDA_GRID), None, cap)
                skipped.update(id=skipped["id"].replace("/nan/", "/no-lambda/"), skipped=True)
                tasks.append(skipped)
                continue
            tasks.append(_task(prefix, dataset, u, arm, lam, grid, None, cap))
    ck = run_tasks([t for t in tasks if not t.get("skipped")], run_dir, prefix, execute_fn, workers,
                   check_busy=check_busy)
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
            if t.get("skipped"):
                ok = False
                continue
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
        base = {"id": tid, "split": t["split"], "fold": t["fold"], "arm": t["arm"], "max_iter": t["max_iter"]}
        if t.get("skipped"):
            rows.append({**base, "readable": False, "skipped": "no lambda with a readable inner fit"})
            continue
        s = ck.done[tid]["summary"]
        rows.append({**base, **{k: s[k] for k in ("lam_requested", "lam", "readable", "refit", "at_cap",
                                                  "n_iter", "loss", "data_loss", "n_params", "seconds",
                                                  "moved", "shares")}})
    return rows


def cap_hits(run_dir: Path) -> dict:
    """Every fit's cap-hit, from every checkpoint: per stage and arm family, fits and cap-hits."""
    out: dict[str, dict] = {}
    for path in sorted((Path(run_dir) / "checkpoints").rglob("*.jsonl")):
        stage = str(path.relative_to(Path(run_dir) / "checkpoints").with_suffix(""))
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            summary = record.get("summary", {})
            if "at_cap" not in summary:
                continue
            arm = record["id"].split("/")[-4]  # .../<arm>/<lambda>/it<cap>/<tag>
            key = f"{stage}|{P.family_of(arm)}"
            cell = out.setdefault(key, {"fits": 0, "at_cap": 0, "at_cap_ids": []})
            cell["fits"] += 1
            if summary["at_cap"]:
                cell["at_cap"] += 1
                cell["at_cap_ids"].append(record["id"])
    return out


def run_dataset(run_dir: Path, prefix: str, dataset: str, arms_by_unit: dict, layout,
                execute_fn=execute, workers: int = C.WORKERS, check_busy: bool = True,
                caps: dict | None = None) -> dict:
    """Selection, finals and pooling for one data set; resumable from its checkpoint."""
    choice = select_lambdas(run_dir, prefix, dataset, arms_by_unit, execute_fn, workers,
                            check_busy=check_busy, caps=caps)
    ck, tasks = final_fits(run_dir, prefix, dataset, arms_by_unit, choice, execute_fn, workers,
                           check_busy=check_busy, caps=caps)
    T, _ = load_dataset(Path(run_dir), dataset)
    return {"choice": choice, "pooled": pool(run_dir, ck, tasks, T, layout),
            "fits": fit_summaries(ck, tasks)}

