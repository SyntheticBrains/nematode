"""Section 8's readings (R-0 to R-D) from pooled predictions, and section 9's branch mapper.

score_split works on any target (S, or an oracle's S_syn): it is the scoring step only, and the
caller decides what the target is. Only ceiling.scorer passes S.
"""

from __future__ import annotations


import numpy as np

from ceiling import config as C
from ceiling.bootstrap import interval, r_point, resampled_r
from ceiling.splits import Layout, unit_class

REFERENCES = ("floor-C", "H0-cal", "C.2-cal")

# ---- Scoring ----------------------------------------------------------------------------------


def score_split(target: np.ndarray, preds: dict[str, np.ndarray | None], entries: np.ndarray,
                split: str, layout: Layout, counts: np.ndarray,
                fold_entries: list[np.ndarray] | None = None) -> dict[str, dict]:
    """Every arm's point r (nan_corr over the entry set), its resampled r and per-fold r.
    An arm whose prediction is None is unreadable on the split."""
    rows, cols = np.nonzero(entries)
    y = target[rows, cols]
    unit = unit_class(split, layout, rows, cols)
    out: dict[str, dict] = {}
    for arm, pred in preds.items():
        if pred is None:
            out[arm] = {"readable": False}
            continue
        vec = pred[rows, cols]
        per_fold = [r_point(target[f], pred[f]) for f in (fold_entries or [])]
        out[arm] = {"readable": True, "r": r_point(y, vec), "n": int(np.isfinite(vec).sum()),
                    "boot": resampled_r(y, vec, unit, counts), "per_fold": per_fold}
    return out


def contrast(a: dict, b: dict | float) -> dict | None:
    """a - b, paired: b is an arm's scores, or a fixed number."""
    if not a.get("readable") or (isinstance(b, dict) and not b.get("readable")):
        return None
    if isinstance(b, dict):
        diff, boot = a["r"] - b["r"], a["boot"] - b["boot"]
    else:
        diff, boot = a["r"] - b, a["boot"] - b
    return {"diff": float(diff), "ci": interval(boot)}


def family(scores: dict[str, dict], masked: str, draws: list[str]) -> dict:
    """masked against a family: readable draws, rank, and masked - median, paired."""
    readable = [d for d in draws if scores.get(d, {}).get("readable")]
    out = {"draws": len(draws), "readable_draws": len(readable)}
    if not readable or not scores[masked].get("readable"):
        return out
    rs = np.array([scores[d]["r"] for d in readable])
    r_masked = scores[masked]["r"]
    beaten_by = int(np.sum(rs >= r_masked)) if C.RANK_STRICT else int(np.sum(rs > r_masked))
    median_boot = np.median(np.stack([scores[d]["boot"] for d in readable]), axis=0)
    out.update({"rank": 1 + beaten_by, "n_ranked": 1 + len(readable), "median": float(np.median(rs)),
                "diff": float(r_masked - np.median(rs)),
                "ci": interval(scores[masked]["boot"] - median_boot),
                "values": {d: scores[d]["r"] for d in readable}})
    return out


def minimum_effect(scores: dict[str, dict], references: tuple[str, ...] = REFERENCES) -> dict:
    """m = max(0.25 (r_masked - r_ref), 0.0109), r_ref the best reference on the split."""
    refs = {k: scores[k]["r"] for k in references if scores.get(k, {}).get("readable")}
    if not refs or not scores.get("masked", {}).get("readable"):
        return {"m": float("nan"), "r_ref": float("nan"), "ref_arm": None}
    ref_arm = max(refs, key=refs.get)
    m = max(C.M_FRACTION * (scores["masked"]["r"] - refs[ref_arm]), C.M0)
    return {"m": float(m), "r_ref": float(refs[ref_arm]), "ref_arm": ref_arm}


def family_passes(fam: dict, m: float) -> bool:
    """R-B's bar: first of the ranked set, at least m above the median, interval above 0."""
    if "rank" not in fam:
        return False
    return fam["rank"] == 1 and fam["diff"] >= m and fam["ci"][0] > 0


# ---- Section 9: branch names, verbatim -----------------------------------------------------------

RA_BELOW = "below the reference"
RA_NOT_ABOVE = "not above the reference"
RA_ABOVE_UNDER_M = "above, under the minimum"
RA_ABOVE = "above the reference"
GATE_K = "unreadable: the channel is not identified for this family"
UNREADABLE = "Unreadable"
BIASED = "Structurally biased"
REVERSED = "Reversed"
RB_TABLE = {
    (True, True): "Anatomy helps beyond degree and reciprocity statistics.",
    (True, False): "The advantage is reciprocity and overlap",
    (False, True): "Anatomy beats reciprocity-matched rewirings, but not degree-matched ones.",
    (False, False): "Anatomy does not help a learned model here.",
}
RC_TABLE = {
    (True, True): "Anatomy helps at equal size, beyond random sparsity.",
    (False, True): "Anatomy beats random sparsity",
    (True, False): "Sparsity helps, not anatomy",
    (False, False): "Neither the constraint nor sparsity helps here.",
}
DENOISES = "denoises beyond copying the training matrix"
NOT_DENOISES = "unregistered: does not denoise beyond copying the training matrix"
RD_ABOVE = "Above the attainable level"
RD_DATA = "Data-limited"
RD_MODEL = "Model- or truth-limited"
RD_UNRESOLVED = "Unresolved"
RD_GAP = "unregistered: above r_att by more than m, interval includes 0"
DESCRIPTION = "description only (the split's oracles were cut)"
NOT_READ_K_STIM = "not read: oracle K failed on held-out stimulations, so only held-out worms is read"
GATE_P = f"{UNREADABLE}: oracle P failed on the split"


def _ra(cell: dict | None, m: float) -> str:
    if cell is None:
        return f"{UNREADABLE}: the arm is unreadable"
    lo, hi = cell["ci"]
    if hi < 0:
        return RA_BELOW
    if lo <= 0:
        return RA_NOT_ABOVE
    return RA_ABOVE if cell["diff"] >= m else RA_ABOVE_UNDER_M


def _rb(split: dict, m: float, k_ok: bool, p_ok: bool, n_biased: bool) -> str:
    if not split["masked_readable"] or not k_ok or not p_ok:
        return UNREADABLE
    if n_biased:
        return BIASED
    fams = split["R-B"]
    readable = {f: fams[f]["readable_draws"] >= C.NULL_READABLE_MIN for f in ("null-D", "null-R")}
    live = [f for f in ("null-D", "null-R") if readable[f]]
    if not live:
        return f"{UNREADABLE}: both families unreadable"
    if any(fams[f]["ci"][1] < -m for f in live):
        return REVERSED
    passes = {f: family_passes(fams[f], m) for f in live}
    if len(live) == 1:
        other = "null-R" if live[0] == "null-D" else "null-D"
        verdict = "passes" if passes[live[0]] else "fails"
        return f"{other} unreadable; {live[0]} read alone: {verdict}"
    return RB_TABLE[(passes["null-D"], passes["null-R"])]


def _won(cell: dict | None, m: float) -> bool:
    return cell is not None and cell["diff"] >= m and cell["ci"][0] > 0


def _rc(split: dict, m: float, p_worm_ok: bool) -> str:
    if not p_worm_ok or not split["masked_readable"]:
        return UNREADABLE
    cells = split["R-C"]
    tied, er = cells.get("tied"), cells.get("sparse-ER")
    if any(c is not None and c["ci"][1] < -m for c in (tied, er)):
        return REVERSED
    return RC_TABLE[(_won(tied, m), _won(er, m))]


def _rd(cell: dict | None, m: float) -> str:
    if cell is None:
        return f"{UNREADABLE}: the arm is unreadable"
    d, (lo, hi) = cell["diff"], cell["ci"]
    if abs(d) <= m:
        return RD_DATA
    if d > m:
        return RD_ABOVE if lo > 0 else RD_GAP
    if hi < 0:
        return RD_MODEL
    return RD_UNRESOLVED


def map_branches(result: dict) -> dict:
    """Section 9's registered branch for every reading, with its precedence.

    `result` holds "oracles" (K and P per split, P's held-out-worm line, N's bias per split),
    optional "description_only" splits, and per split its m, masked readability and the
    contrasts (see the tests for the shape).
    """
    oracles, splits = result["oracles"], result["splits"]
    description_only = set(result.get("description_only", ()))
    k_stim_ok = bool(oracles["K"]["stimulations"])
    out: dict = {}
    read: set[str] = set()  # held-out splits whose readings were made (not gated, not description)
    for name in C.HELD_OUT_SPLITS:
        split = splits.get(name)
        if name in description_only:
            out[name] = {"R-A": DESCRIPTION, "R-B": DESCRIPTION, "R-D": DESCRIPTION}
            continue
        gate = None
        if not oracles["K"][name]:
            gate = GATE_K
        elif name != "stimulations" and not k_stim_ok and C.K_STIM_FAIL_BLOCKS_NEURONS:
            gate = NOT_READ_K_STIM
        elif not oracles["P"][name]:
            gate = GATE_P
        if gate is not None:
            out[name] = {"R-A": {"masked": gate, "tied": gate},
                         "R-B": UNREADABLE if gate == GATE_P else gate, "R-D": gate}
            continue
        if split is None:
            raise ValueError(f"no readings for {name}, and no gate explains why")
        m = split["m"]
        read.add(name)
        out[name] = {
            "R-A": {arm: _ra(split["R-A"].get(arm), m) for arm in ("masked", "tied")},
            "R-B": _rb(split, m, True, True, bool(oracles["N"][name])),
            "R-D": _rd(split.get("R-D"), m),
        }
    worms = splits["worms"]
    p_worm = bool(oracles["P"]["worms"])
    denoise = worms["R-C"].get("denoising")
    out["worms"] = {
        "R-C": _rc(worms, worms["m"], p_worm),
        "denoising": (UNREADABLE if not p_worm or not worms["masked_readable"]
                      else DENOISES if _won(denoise, worms["m"]) else NOT_DENOISES),
    }
    out["governing"] = "stimulations"
    out["disagreements"] = []
    if read == set(C.HELD_OUT_SPLITS):  # named, not resolved; only between two read splits
        stim, neur = out["stimulations"], out["neurons"]
        out["disagreements"] = ([k for k in ("R-B", "R-D") if neur[k] != stim[k]]
                                + [f"R-A/{a}" for a in ("masked", "tied") if neur["R-A"][a] != stim["R-A"][a]])
    return out

