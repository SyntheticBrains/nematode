"""Readings on one target from pooled predictions, and the results files.

The target is S only when ceiling.scorer calls this; the oracles call it with their S_syn.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ceiling import config as C
from ceiling import plan as P
from ceiling.bootstrap import interval, resample_counts
from ceiling.readings import REFERENCES, contrast, family, map_branches, minimum_effect, score_split


def fold_entry_masks(entry, split: str) -> list[np.ndarray]:
    n = 1 if split == "worms" else C.N_FOLDS
    return [np.asarray(entry[f"{split}/{f}"]) for f in range(n)]


def split_scores(target, preds: dict, entry, split: str, layout, counts) -> dict:
    folds = fold_entry_masks(entry, split)
    entries = np.any(np.stack(folds), axis=0)
    return score_split(target, preds, entries, split, layout, counts, folds if split != "worms" else None)


def _strip(scores: dict) -> dict:
    out = {}
    for arm, s in scores.items():
        if not s.get("readable"):
            out[arm] = {"readable": False}
            continue
        out[arm] = {"readable": True, "r": s["r"], "ci": interval(s["boot"]), "n": s["n"],
                    "per_fold": s["per_fold"]}
    return out


def split_readings(scores: dict, split: str, plan: dict, references: tuple[str, ...],
                   r_att: float | None = None) -> dict:
    """m and every registered contrast on one split (whichever arms are present)."""
    m = minimum_effect(scores, references)
    out: dict = {"m": m["m"], "r_ref": m["r_ref"], "ref_arm": m["ref_arm"],
                 "masked_readable": bool(scores.get("masked", {}).get("readable"))}
    ref = scores.get(m["ref_arm"]) if m["ref_arm"] else None
    out["R-0"] = {"C.2 - H0": contrast(scores["C.2"], scores["H0"])
                  if "C.2" in scores and "H0" in scores else None}
    if split in C.HELD_OUT_SPLITS:
        out["R-A"] = {arm: (contrast(scores[arm], ref) if ref and arm in scores else None)
                      for arm in ("masked", "tied")}
        out["R-B"] = {"null-D": family(scores, "masked", [a for a in plan["null-D"] if a in scores]),
                      "null-R": family(scores, "masked", [a for a in plan["null-R"] if a in scores])}
        if r_att is not None and math.isfinite(r_att) and "masked" in scores:
            out["R-D"] = contrast(scores["masked"], r_att) | {"r_att": r_att}
        else:
            out["R-D"] = None
    else:
        er = family(scores, "masked", [a for a in plan["sparse-ER"] if a in scores])
        out["R-C"] = {
            "tied": contrast(scores["masked"], scores["tied"]) if "tied" in scores else None,
            "sparse-ER": ({"diff": er["diff"], "ci": er["ci"], "median": er["median"]}
                          if "diff" in er else None),
            "denoising": contrast(scores["masked"], scores["T"]) if "T" in scores else None,
        }
    return out


def readings_on_target(target, T, layout, pooled: dict, plan: dict, oracle: dict | None, entry,
                       is_real: bool) -> dict:
    counts = resample_counts(layout.n_classes)
    references = REFERENCES if is_real else ("floor-C",)
    results: dict = {"splits": {}, "scores": {}}
    for split in [x for x in C.SPLITS if x in pooled]:
        preds = dict(pooled[split])
        if split == "worms":
            preds["T"] = T
        scores = split_scores(target, preds, entry, split, layout, counts)
        r_att = None
        if is_real and oracle and split in C.HELD_OUT_SPLITS:
            r_att = oracle.get("r_att", {}).get(split, {}).get("r_att")
        results["splits"][split] = split_readings(scores, split, plan, references, r_att)
        results["scores"][split] = _strip(scores)
    if is_real and oracle is not None:
        results["branches"] = map_branches({
            "oracles": oracle["gates"], "description_only": oracle.get("description_only", []),
            "splits": results["splits"]})
    return results


def write_results(results: dict, out_dir: Path, stem: str = "results") -> None:
    out_dir = Path(out_dir)
    (out_dir / f"{stem}.json").write_text(json.dumps(results, indent=1, default=float))
    with open(out_dir / f"{stem}_per_arm.csv", "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["split", "arm", "family", "readable", "r", "ci_low", "ci_high", "n"]
                        + [f"fold_{f}" for f in range(C.N_FOLDS)])
        for split, scores in results["scores"].items():
            for arm, s in scores.items():
                folds = s.get("per_fold", []) if s.get("readable") else []
                row = [split, arm, P.family_of(arm), int(bool(s.get("readable")))]
                if s.get("readable"):
                    row += [repr(s["r"]), repr(s["ci"][0]), repr(s["ci"][1]), s["n"]]
                else:
                    row += ["", "", "", ""]
                row += [repr(v) for v in folds] + [""] * (C.N_FOLDS - len(folds))
                writer.writerow(row)
