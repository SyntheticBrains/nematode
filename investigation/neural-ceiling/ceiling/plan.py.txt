"""What is fitted where: the arms of each data set and unit, their lambda groups, the fit counts,
and the cost projection with section 10's cuts.
"""

from __future__ import annotations

from ceiling import config as C

NULL_D = [f"nullD:{s}" for s in C.SEEDS_NULL_D]
NULL_R = [f"nullR:{s}" for s in C.SEEDS_NULL_R]
ER = [f"er:{s}" for s in C.SEEDS_ER]
PRIOR = ["prior:masked"] + [f"prior:{a}" for a in NULL_D + NULL_R]
SWEEP = [f"tied:{r}" for r in C.TIED_SWEEP_RANKS]
TIED = "tied"
DENSE = "dense"
UNTIED = f"untied:{C.UNTIED_RANK}"
SELECTION_SET = (["masked"] + [f"nullD:{s}" for s in C.SELECTION_NULL_D]
                 + [f"nullR:{s}" for s in C.SELECTION_NULL_R])
FAMILIES = {"null-D": NULL_D, "null-R": NULL_R, "sparse-ER": ER}


def support_of(arm: str) -> str:
    """The support an arm is fitted on ("masked", "nullD:<seed>", ...)."""
    return arm.removeprefix("prior:")


def family_of(arm: str) -> str:
    if arm.startswith("prior:"):
        return "masked-prior"
    head = arm.split(":")[0]
    return {"masked": "masked", "nullD": "null-D", "nullR": "null-R", "er": "sparse-ER",
            "tied": "tied-sweep" if ":" in arm else "tied", "dense": "dense",
            "untied": "untied"}.get(head, "reference")


def group_of(arm: str) -> str:
    """Arms sharing a lambda: every sparse-support arm; otherwise the arm itself."""
    if family_of(arm) in ("masked", "null-D", "null-R", "sparse-ER", "masked-prior"):
        return "sparse"
    return arm


def selection_arms(group: str) -> list[str]:
    return list(SELECTION_SET) if group == "sparse" else [group]


def caps(long_cap: int) -> dict[str, int]:
    """Iteration caps: the long cap for sparse-support and tied fits, the registered 500 for the
    rest (amendment 1, item 2; long_cap is 500 unless the preflight's convergence check raises it)."""
    return {"default": C.MAX_ITER, "long": long_cap}


def cap_of(arm: str, caps: dict[str, int] | None) -> int:
    if caps is None:
        return C.MAX_ITER
    return caps["long"] if family_of(arm) in C.LONG_CAP_FAMILIES else caps["default"]


def long_cap_kinds() -> tuple[str, ...]:
    """The timing kinds whose fits take the long cap."""
    return ("sparse", "sparse_nullD", "prior", TIED)


def kind_of(arm: str) -> str:
    """The timing kind of an arm's fit."""
    fam = family_of(arm)
    if fam in ("masked", "null-R", "sparse-ER"):
        return "sparse"
    if fam == "null-D":
        return "sparse_nullD"
    if fam == "masked-prior":
        return "prior"
    return arm  # tied, dense, untied:17, tied:<r>


def units(splits=C.SPLITS) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    for split in splits:
        out += [(split, None)] if split == "worms" else [(split, f) for f in range(C.N_FOLDS)]
    return out


# ---- arms per data set ---------------------------------------------------------------------------


def real_arms(split: str, cuts: tuple[str, ...] = ()) -> list[str]:
    arms = ["masked"] + NULL_D + NULL_R + ER + [TIED]
    if "description_arms" not in cuts:
        arms += PRIOR + [DENSE] + SWEEP + ([UNTIED] if split == "worms" else [])
    return arms


def p_arms(split: str) -> list[str]:
    arms = ["masked"] + NULL_D + NULL_R
    return arms + ER + [TIED] if split == "worms" else arms


def n_arms(generating: str) -> list[str]:
    if not C.N_SELECTION_INCLUDES_GENERATING:
        raise NotImplementedError("N's selection set without its generating draw")
    return ["masked"] + [d for d in NULL_D if d != generating]


def oracle_splits(cuts: tuple[str, ...]) -> tuple[str, ...]:
    return ("stimulations",) if "neuron_split_oracles" in cuts else C.HELD_OUT_SPLITS


def n_truths(cuts: tuple[str, ...]) -> tuple[int, ...]:
    return C.N_TRUTH_SEEDS[:2] if "n_two_truths" in cuts else C.N_TRUTH_SEEDS


def datasets(cuts: tuple[str, ...] = ()) -> dict[str, dict]:
    """Every data set's units and the arms of each unit."""
    held = oracle_splits(cuts)
    out = {"real": {u: real_arms(u[0], cuts) for u in units()}}
    for seed in C.SEEDS_P:
        out[f"P{seed}"] = {u: p_arms(u[0]) for u in units(held + ("worms",))}
    for seed in n_truths(cuts):
        out[f"N{seed}"] = {u: n_arms(f"nullD:{seed}") for u in units(held)}
    out["K"] = {u: ["masked"] for u in units(held)}
    return out


# ---- fit counts and cost -------------------------------------------------------------------------


def unit_fits(arms: list[str], selection: bool = True, n_grid: int = len(C.LAMBDA_GRID),
              n_inner: int = C.N_INNER) -> dict[str, int]:
    """Fits per timing kind for one unit: the lambda selection of each group, then the finals."""
    counts: dict[str, int] = {}
    if selection:
        for group in sorted({group_of(a) for a in arms}):
            for arm in selection_arms(group):
                counts[kind_of(arm)] = counts.get(kind_of(arm), 0) + n_grid * n_inner
    for arm in arms:
        counts[kind_of(arm)] = counts.get(kind_of(arm), 0) + 1
    return counts


def fit_counts(cuts: tuple[str, ...] = ()) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name, plan in datasets(cuts).items():
        total: dict[str, int] = {}
        for arms in plan.values():
            for kind, k in unit_fits(arms, selection=name != "K").items():
                key = "K" if name == "K" else kind
                total[key] = total.get(key, 0) + k
        out[name] = total
    truths = 1 + len(n_truths(cuts))  # M* and N's truths: own inner CV, then one fit
    out["truths"] = {"sparse": len(C.LAMBDA_GRID) * C.N_INNER + 1,
                     "sparse_nullD": (truths - 1) * (len(C.LAMBDA_GRID) * C.N_INNER + 1)}
    return out


def projection(seconds_by_kind: dict[str, float], cuts: tuple[str, ...] = (),
               workers: int = C.WORKERS) -> dict:
    """Serial seconds and hours on `workers` processes, per data set and in all."""
    counts = fit_counts(cuts)
    per: dict[str, dict] = {}
    for name, kinds in counts.items():
        missing = [k for k in kinds if k not in seconds_by_kind]
        if missing:
            raise KeyError(f"no timing for {missing}")
        serial = sum(n * seconds_by_kind[k] for k, n in kinds.items())
        per[name] = {"fits": sum(kinds.values()), "serial_s": serial}
    serial = sum(v["serial_s"] for v in per.values())
    return {"cuts": list(cuts), "per_dataset": per, "fits": sum(v["fits"] for v in per.values()),
            "serial_hours": serial / 3600, "hours_on_workers": serial / 3600 / workers,
            "workers": workers}


def cut_order(seconds_by_kind: dict[str, float], budget_hours: float = C.BUDGET_HOURS) -> dict:
    """Apply section 10's cuts one at a time until the projection is within the budget."""
    cuts: list[str] = []
    steps = [projection(seconds_by_kind, ())]
    for cut in C.CUTS:
        if steps[-1]["hours_on_workers"] <= budget_hours:
            break
        cuts.append(cut)
        steps.append(projection(seconds_by_kind, tuple(cuts)))
    over = steps[-1]["hours_on_workers"] > budget_hours
    return {"cuts": cuts, "steps": steps, "within_budget": not over,
            "action": "stop and amend" if over else "proceed"}
