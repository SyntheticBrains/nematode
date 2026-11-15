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
    An arm whose prediction is None is unreadable on the split. A readable arm must be defined on
    every entry of the set, so every arm is scored on the same entries (amendment 1, item 5)."""
    rows, cols = np.nonzero(entries)
    y = target[rows, cols]
    unit = unit_class(split, layout, rows, cols)
    out: dict[str, dict] = {}
    for arm, pred in preds.items():
        if pred is None:
            out[arm] = {"readable": False}
            continue
        vec = pred[rows, cols]
        defined = int(np.isfinite(vec).sum())
        if defined != len(vec):
            raise AssertionError(f"{split}: {arm} is defined on {defined} of the {len(vec)} entries")
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
    if not readable or not scores.get(masked, {}).get("readable"):
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
NOT_DENOISES = "does not denoise beyond copying"  # amendment 1 item 13, IMPLEMENTATION_NOTES 21
RD_ABOVE = "Above the attainable level"
RD_DATA = "Data-limited"
RD_MODEL = "Model- or truth-limited"
RD_UNRESOLVED = "Unresolved"
RD_GAP = "above, unresolved"  # amendment 1 item 13, IMPLEMENTATION_NOTES 21
DESCRIPTION = "description only (the split's oracles were cut)"
NOT_READ_K_STIM = "not read: oracle K failed on held-out stimulations, so only held-out worms is read"
GATE_P = f"{UNREADABLE}: oracle P failed on the split"
ARM_UNREADABLE = f"{UNREADABLE}: the arm is unreadable"
M_UNDEFINED = f"{UNREADABLE}: m is undefined while masked is unreadable"


def _ra(cell: dict | None, m: float) -> str:
    if cell is None:
        return ARM_UNREADABLE
    if not m == m:  # m needs masked's r
        return M_UNDEFINED
    lo, hi = cell["ci"]
    if hi < 0:
        return RA_BELOW
    if lo <= 0:
        return RA_NOT_ABOVE
    return RA_ABOVE if cell["diff"] >= m else RA_ABOVE_UNDER_M


def _rb(split: dict, m: float, n_biased: bool) -> str:
    """R-B after its K and P gates: unreadable masked, then N, then readability, reversal, table."""
    if not split["masked_readable"]:
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
    if len(live) == 1:  # amendment 1 item 13, IMPLEMENTATION_NOTES 21
        return f"{live[0]} read alone: {'passes' if passes[live[0]] else 'fails'}"
    return RB_TABLE[(passes["null-D"], passes["null-R"])]


def _won(cell: dict | None, m: float) -> bool:
    return cell is not None and cell["diff"] >= m and cell["ci"][0] > 0


def _rc(split: dict, m: float) -> str:
    """R-C after its gate. An unreadable comparator makes its contrast unreadable, never "fails"
    (amendment 1, item 7): tied unreadable, or fewer than 4 of 5 sparse-ER draws readable."""
    if not split["masked_readable"]:
        return UNREADABLE
    cells = split["R-C"]
    live = {name: cells.get(name) for name in ("tied", "sparse-ER") if cells.get(name) is not None}
    if not live:
        return f"{UNREADABLE}: both contrasts unreadable"
    if any(c["ci"][1] < -m for c in live.values()):
        return REVERSED
    if len(live) == 1:
        (name, cell), = live.items()
        return f"{name} read alone: {'wins' if _won(cell, m) else 'fails'}"
    return RC_TABLE[(_won(live["tied"], m), _won(live["sparse-ER"], m))]


def _rd(cell: dict | None, m: float) -> str:
    if cell is None:
        return ARM_UNREADABLE
    d, (lo, hi) = cell["diff"], cell["ci"]
    if abs(d) <= m:
        return RD_DATA
    if d > m:
        return RD_ABOVE if lo > 0 else RD_GAP
    if hi < 0:
        return RD_MODEL
    return RD_UNRESOLVED


def _gate(reading: str, name: str, oracles: dict) -> str | None:
    """The first of a reading's gates that fails on the split, as its registered label."""
    for gate in C.GATES[reading]:
        if gate == "K":
            if not oracles["K"][name]:
                return GATE_K
            if name != "stimulations" and not oracles["K"]["stimulations"] and C.K_STIM_FAIL_BLOCKS_NEURONS:
                return NOT_READ_K_STIM
        elif gate == "P" and not oracles["P"][name]:
            return UNREADABLE if reading == "R-B" else GATE_P
        elif gate == "P-worms" and not oracles["P"]["worms"]:
            return UNREADABLE
    return None


def map_branches(result: dict) -> dict:
    """Section 9's registered branch for every reading, with its precedence and amendment 1's gates
    (item 6): R-A by K; R-B by K, P and N; R-D by K and P; R-C and denoising by P's held-out-worm
    line. A split whose masked fit is unreadable reads as unreadable.

    `result` holds "oracles" (K and P per split, P's held-out-worm line, N's bias per split),
    optional "description_only" splits, and per split its m, masked readability and the
    contrasts (see the tests for the shape).
    """
    oracles, splits = result["oracles"], result["splits"]
    description_only = set(result.get("description_only", ()))
    out: dict = {}
    read: dict[str, set[str]] = {}  # the readings each held-out split actually made
    for name in C.HELD_OUT_SPLITS:
        if name in description_only:
            out[name] = {"R-A": DESCRIPTION, "R-B": DESCRIPTION, "R-D": DESCRIPTION}
            continue
        split = splits.get(name)
        cells, made = {}, set()
        for reading in ("R-A", "R-B", "R-D"):
            gate = _gate(reading, name, oracles)
            if gate is not None:
                cells[reading] = {"masked": gate, "tied": gate} if reading == "R-A" else gate
                continue
            if split is None:
                raise ValueError(f"no readings for {name}, and no gate explains why")
            m = split["m"]
            if reading == "R-A":
                cells[reading] = {arm: _ra(split["R-A"].get(arm), m) for arm in ("masked", "tied")}
            elif reading == "R-B":
                cells[reading] = _rb(split, m, bool(oracles["N"][name]))
            else:
                cells[reading] = _rd(split.get("R-D"), m)
            made.add(reading)
        out[name], read[name] = cells, made
    worms = splits["worms"]
    gate_rc, gate_dn = _gate("R-C", "worms", oracles), _gate("denoising", "worms", oracles)
    denoise = worms["R-C"].get("denoising")
    out["worms"] = {
        "R-C": gate_rc or _rc(worms, worms["m"]),
        "denoising": gate_dn or (UNREADABLE if not worms["masked_readable"] or denoise is None
                                 else DENOISES if _won(denoise, worms["m"]) else NOT_DENOISES),
    }
    out["governing"] = "stimulations"
    out["disagreements"] = []  # named, not resolved; only where both held-out splits made the reading
    both = read.get("stimulations", set()) & read.get("neurons", set())
    stim, neur = out.get("stimulations", {}), out.get("neurons", {})
    for reading in ("R-A", "R-B", "R-D"):
        if reading not in both:
            continue
        if reading == "R-A":
            out["disagreements"] += [f"R-A/{a}" for a in ("masked", "tied") if neur["R-A"][a] != stim["R-A"][a]]
        elif neur[reading] != stim[reading]:
            out["disagreements"].append(reading)
    return out
