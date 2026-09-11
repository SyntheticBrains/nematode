"""Re-read every committed table under the mixture-aware family.

Descriptive, and deliberately powerless: **a committed verdict is not changed here.** Each stands
as registered, in the units and under the rule it was registered with; this places a second,
pre-specified reading beside it and is an input to the ladder re-read and to nothing else. Where
the two disagree, both are reported and the registered one is the verdict.

Every table is read from its committed per-seed CSV rather than from a campaign directory, which
Logbook 042 recorded as the practice that made its analysis reproducible from the repository.

Two protocols, not pooled:

* **panels** contrast one arm against another arm, paired by seed;
* **assays** contrast one arm against each seed's own committed comparator, carried in the table's
  own ``frozen_clone`` column.

The graded metric is available for panels only: the panel tables carry ``foods`` beside ``success``
and the assay tables do not. That is a property of what was committed, and the record says so
rather than quietly reading fewer tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import l4_mixture_statistic as ms  # noqa: E402  # pyright: ignore[reportMissingImports]
from weight_search_architecture_ranking import (  # noqa: E402
    bh_fdr,  # pyright: ignore[reportMissingImports]
)

REPO = _HERE.parent.parent
SUPPORTING = REPO / "docs" / "experiments" / "logbooks" / "supporting"
COMPARATOR_COLUMN = "frozen_clone"


class Contrast(NamedTuple):
    """One registered comparison within one committed table."""

    table: str
    name: str
    arm_a: str
    arm_b: str | None  # None: contrast against the table's own per-seed comparator column
    note: str


# The headline contrast of each committed table, named as its logbook named it. An assay's
# ``arm_b`` is None: its comparator is the per-seed committed clone in its own table.
CONTRASTS: tuple[Contrast, ...] = (
    Contrast("040-l4-panel", "wiring_plastic", "wt_plastic", "rn_plastic", "panel 1's T1"),
    Contrast(
        "040-l4-panel",
        "wiring_hebbian",
        "wt_hebbian",
        "rn_hebbian",
        "the Hebbian wiring contrast on panel 1's seeds; not itself a registered test there",
    ),
    Contrast(
        "041-l4-panel2",
        "wiring_hebbian",
        "wt_hebbian",
        "rn_hebbian",
        "panel 2's P1, the primary",
    ),
    Contrast(
        "041-l4-panel2",
        "wiring_hebbian_count",
        "wt_hebbian_count",
        "rn_hebbian_count",
        "panel 2's P2, count-initialised",
    ),
    Contrast("042-l4-panel3", "wiring_hebbian", "wt_hebbian", "rn_hebbian", "panel 3's R1"),
    Contrast(
        "044-l4-atlas-signs",
        "wiring_hebbian_atlas",
        "wt_hebbian_atlas",
        "rn_hebbian_atlas",
        "grounded signs, G2",
    ),
    Contrast(
        "044-l4-atlas-signs",
        "wiring_hebbian_dale",
        "wt_hebbian_dale",
        "rn_hebbian_dale",
        "Dale's law enforced, G4",
    ),
    Contrast("045-l4-consolidation", "anchor", "anchor", None, "clone assay"),
    Contrast("045-l4-consolidation", "rigidity", "rigidity", None, "clone assay"),
    Contrast("045-l4-consolidation", "oracle", "oracle", None, "clone assay"),
    Contrast("046-l4-decorrelation", "wiring_antihebb", "wt_antihebb", "rn_antihebb", "S1"),
    Contrast("046-l4-decorrelation", "wiring_oja", "wt_oja", "rn_oja", "S2"),
    Contrast("047-l4-structured-instruction", "routing_wt", "wt_pathway", "wt_global", "S1"),
    Contrast("047-l4-structured-instruction", "routing_rn", "rn_pathway", "rn_global", "S2"),
    Contrast(
        "050-l4-perturbation-clone-assay",
        "node_perturbation",
        "node_perturbation",
        None,
        "clone assay",
    ),
    Contrast(
        "050-l4-perturbation-clone-assay",
        "perturbation_frozen",
        "perturbation_frozen",
        None,
        "its frozen control",
    ),
    Contrast(
        "052-l4-endpoint-evaluation",
        "endpoint_nodeperturbation",
        "endpoint_nodeperturbation",
        None,
        "endpoint, perturbation off",
    ),
)


def read_table(table: str) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    """Per-arm success and foods from a committed per-seed table, keyed by seed."""
    path = SUPPORTING / table / "per-seed.csv"
    success: dict[str, dict[int, float]] = {}
    foods: dict[str, dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            arm, seed = row["arm"], int(row["seed"])
            success.setdefault(arm, {})[seed] = float(row["success"])
            if row.get("foods"):
                foods.setdefault(arm, {})[seed] = float(row["foods"])
            if row.get(COMPARATOR_COLUMN):
                success.setdefault(COMPARATOR_COLUMN, {})[seed] = float(row[COMPARATOR_COLUMN])
    return success, foods


def _values(
    arms: dict[str, dict[int, float]],
    contrast: Contrast,
) -> tuple[dict[int, float], dict[int, float]] | None:
    """Return the two per-seed series a contrast compares, or None where the table lacks them."""
    a = arms.get(contrast.arm_a)
    b = arms.get(contrast.arm_b) if contrast.arm_b else arms.get(COMPARATOR_COLUMN)
    if not a or not b:
        return None
    return a, b


def reread(contrast: Contrast) -> dict[str, Any]:
    """Read one committed contrast under the family, on both metrics where both exist."""
    success, foods = read_table(contrast.table)
    out: dict[str, Any] = {
        "table": contrast.table,
        "contrast": contrast.name,
        "arm_a": contrast.arm_a,
        "arm_b": contrast.arm_b or f"{COMPARATOR_COLUMN} (per-seed comparator)",
        "protocol": "panel" if contrast.arm_b else "assay",
        "note": contrast.note,
    }
    primary = _values(success, contrast)
    if primary is None:
        out["error"] = "arms not present in the committed table"
        return out
    out["primary"] = ms.read(*primary)
    graded = _values(foods, contrast)
    out["graded"] = (
        _graded(*graded, primary[0], primary[1])
        if graded is not None
        else {"available": False, "reason": "the committed table carries no foods column"}
    )
    return out


def _graded(
    a_foods: dict[int, float],
    b_foods: dict[int, float],
    a_success: dict[int, float],
    b_success: dict[int, float],
) -> dict[str, Any]:
    """Level and shift on the graded metric, over the competence the primary metric defines.

    No threshold is chosen here: a seed is competent if its FULL-CLEAR value clears the committed
    one, and the graded values of those seeds are what the level contrast compares. Choosing a
    foods threshold now, with the outcomes known, is the move the family exists to prevent.
    """
    competent_a = {s for s, v in a_success.items() if v >= ms.COMPETENT_THRESHOLD}
    competent_b = {s for s, v in b_success.items() if v >= ms.COMPETENT_THRESHOLD}
    a_level = {s: v for s, v in a_foods.items() if s in competent_a}
    b_level = {s: v for s, v in b_foods.items() if s in competent_b}
    level = (
        # The competent subsets are already selected on the primary metric, so the level contrast
        # must not apply a threshold again -- the committed one is in full-clear percent and these
        # values are foods. Routing these through the primary reader would do exactly that.
        ms.level_contrast(a_level, b_level, threshold=float("-inf"))
        if a_level and b_level
        else {
            "defined": False,
            "reason": "one arm has no seed competent on the primary metric",
            "effect": float("nan"),
        }
    )
    members = {"L": level, "W": ms.shift_contrast(a_foods, b_foods)}
    # Corrected within itself, as a parallel family: the graded reading is not a member of the
    # primary family and must not borrow its correction.
    defined = [m for m in ("L", "W") if members[m].get("defined", True)]
    for direction in ("improve", "degrade"):
        qs = bh_fdr([members[m][f"p_{direction}"] for m in defined])
        for member, q in zip(defined, qs, strict=True):
            members[member][f"q_{direction}"] = float(q)
        for member in ("L", "W"):
            if member not in defined:
                members[member][f"q_{direction}"] = float("nan")
    directions = {
        m: ms._direction(members[m], members[m]["q_improve"], members[m]["q_degrade"])
        for m in ("L", "W")
    }
    return {
        "available": True,
        "competence_from": "the primary metric at the committed threshold",
        "L": members["L"],
        "W": members["W"],
        "directions": directions,
        "alpha": ms.SIG_Q,
    }


def _jsonable(value: object) -> object:
    """Replace not-a-number with null, recursively, so the record is strict JSON."""
    import math

    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print(rows: list[dict[str, Any]]) -> None:
    """Print the re-read, one line per contrast."""
    print("\nRe-read of the committed tables under the mixture-aware family")
    print("  Descriptive. No committed verdict is changed by anything below.\n")
    for row in rows:
        if "error" in row:
            print(f"  {row['table']:32} {row['contrast']:22} -- {row['error']}")
            continue
        primary = row["primary"]
        directions = "".join(primary["directions"][m] for m in ms.MEMBERS)
        level = primary["members"]["L"]
        level_effect = f"{level['effect']:+6.1f}" if level.get("defined") else "   n/a"
        print(
            f"  {row['table']:32} {row['contrast']:22} {row['protocol']:6} "
            f"F/L/W {directions}  L {level_effect}  "
            f"W {primary['members']['W']['effect']:+6.1f}  -> {primary['verdict']}",
        )
        graded = row["graded"]
        if graded["available"]:
            level_text = (
                f"L {graded['L']['effect']:+.2f}{graded['directions']['L']}"
                if graded["L"].get("defined")
                else "L n/a"
            )
            print(
                f"      graded: {level_text}  "
                f"W {graded['W']['effect']:+.2f}{graded['directions']['W']} foods",
            )


def main(argv: list[str] | None = None) -> int:
    """Re-read every committed contrast and write the records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write the re-read as JSON")
    parser.add_argument("--csv", type=Path, help="write the per-contrast table")
    args = parser.parse_args(argv)

    rows = [reread(contrast) for contrast in CONTRASTS]
    _print(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                _jsonable({"contrasts": rows, "threshold": ms.COMPETENT_THRESHOLD}),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["table", "contrast", "protocol", "F", "L", "W", "l_effect", "w_effect", "verdict"],
            )
            for row in rows:
                if "error" in row:
                    continue
                primary = row["primary"]
                level = primary["members"]["L"]
                writer.writerow(
                    [
                        row["table"],
                        row["contrast"],
                        row["protocol"],
                        *(primary["directions"][m] for m in ms.MEMBERS),
                        f"{level['effect']:.4f}" if level.get("defined") else "",
                        f"{primary['members']['W']['effect']:.4f}",
                        primary["verdict"],
                    ],
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
