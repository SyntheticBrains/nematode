#!/usr/bin/env python
"""A.2: where in the learner's operating region the wiring effect holds, and where it does not.

One inherited pin set the sign of a registered primary once already: the width-by-wiring
interaction ran +0.2818 at ``plasticity_rate`` 0.001 and -0.0657 at 0.0001, a three-way of +0.3475
at q = 0.000 on 81 of 96 seeds, and the effect the pin hid was larger than the effect it was pinned
for. So every pinned setting under a Phase 8 contrast is swept for the learner that reads it, and
the wiring effect is reported as a surface over that region rather than as a number at one point.

**Two halves, because a pin swept on one learner is not swept for another.** PPO reads three pins;
the reading learner reads those three and two of its own. ``plasticity_rate`` and ``trace_decay``
are declared on the shared configuration mixin, so a PPO arm may set either and a hundred committed
configs set the first -- and neither is read under PPO. An arm that looks swept and is not is the
failure this panel is most exposed to, which is why the level's reach into its learner is asserted
by test rather than inferred from the configuration having been accepted.

**This module is a manifest builder and a surface reporter, and deliberately nothing more.** The
two committed instruments are not touched; a test asserts both are byte-identical to ``main``.

**The two halves reach them by different doors, and the reason is in the instruments' constants.**
``wiring_premise.EFFICIENCY_ARMS`` is keyed on ``wt_ppo``/``rn_ppo`` and the family around it is
block V's PPO panel, so the PPO half goes through ``wiring_premise.efficiency_contrast`` -- A.1's
path, and the one those arms are entitled to. The reading learner's arms are ``three_factor``;
labelling them ``wt_ppo`` to reach that function would be a mislabel adopted to suit a signature, so
they call ``connectome_structure_efficiency.analyse`` directly with that module's own arm labels.
That is the path L.1b took, the only previous sweep on this learner. Same four metrics, same BH-FDR
family, same verdict rule either way.

**The primary at each level is the interaction with the campaign's own centre**, per the rule that a
manipulation crossed with a structure contrast is read as an interaction and never as a main effect:
a pin that moves both wirings equally is a fact about the pin. What D16 asks for separately is the
SIGN of the wiring gap at each level, and that is the registered trigger for a full crossing. Both
are reported and neither stands in for the other.

**A frozen floor runs wherever a pin moves the untrained prior, and not elsewhere.** The three
construction pins change the arm before any learning, so each of their levels carries its own floor.
``plasticity_rate`` and ``trace_decay`` cannot reach a frozen arm -- it performs no updates -- so the
centre's floor is theirs. That is the same runs read twice, not a missing control, and the
completeness gate knows the difference: four arms at a construction level, two at a learning-only
one.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import connectome_structure_efficiency as eff  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

# ── Seeds ────────────────────────────────────────────────────────────────────────────────────
# Every campaign in the repository has used 1-96, 101-104 is the programme-wide pilot band, A.1
# took 105-108 for its pilot and 129-160 for its panel. 109-128 and 161+ are untouched.
BURNT_SEEDS = frozenset(range(1, 97)) | frozenset(range(101, 109)) | frozenset(range(129, 161))
PILOT_SEEDS = tuple(range(109, 113))
# The pilot is a SUBSET of levels by design -- it confirms the arms run, the gates fire and the cost
# estimate holds, and four seeds cannot estimate a spread whatever it covers. So it takes the centre
# plus the levels most likely to break: both extremes of a pin never varied on this substrate, and
# the readout width at which no committed PPO arm exists at all.
PILOT_LEVELS: dict[str, tuple[str, ...]] = {
    "ppo": ("centre", "d2", "d6", "wide"),
    "reading": ("centre", "d2", "d6", "td05"),
}
SEEDS_BY_HALF: dict[str, tuple[int, ...]] = {
    "ppo": tuple(range(161, 177)),
    "reading": tuple(range(177, 193)),
}

# ── The cell ─────────────────────────────────────────────────────────────────────────────────
# One cell, and the choice is measured rather than assumed. A.1's committed per-seed data puts
# thermal's interaction spread near 1,044 against a 134-episode wiring effect, so its detectable
# interaction ran 1.6 to 3.4 times the observed effect even at 32 seeds; hard350 came in at 0.50 to
# 0.68. A sweep asks whether a sign moved, and on thermal that cannot be answered at any affordable
# size. Thermal runs only at levels where hard350 shows the sign move.
CELL = "hard_food"

# ── Arms ─────────────────────────────────────────────────────────────────────────────────────
# Named for what they do rather than for the learner, because the reading half's learning arms are
# not PPO arms and calling them so is how a mislabel gets into a manifest.
LEARNING_ARMS = ("wt_learn", "rn_learn")
FROZEN_ARMS = ("wt_frozen", "rn_frozen")
ARMS = (*LEARNING_ARMS, *FROZEN_ARMS)

# How this module's arm names reach each instrument. The PPO half hands `wiring_premise` the arm
# names its `EFFICIENCY_ARMS` is keyed on; the reading half hands the efficiency module its own.
_WP_ARM = {
    "wt_learn": "wt_ppo",
    "rn_learn": "rn_ppo",
    "wt_frozen": "wt_frozen",
    "rn_frozen": "rn_frozen",
}
_EFF_ARM = {"wt_learn": eff._WILD, "rn_learn": eff._REWIRED}

# ── Pins ─────────────────────────────────────────────────────────────────────────────────────
# A pin that changes the arm before any learning; each of its levels needs its own frozen floor.
CONSTRUCTION_PINS = ("readout_width", "forward_pass_depth", "initial_log_std")

CENTRE = "centre"

# (suffix, value) per level. The log-std suffixes are sign-explicit, departing from the committed
# `_ls05` (which means -0.5): this sweep needs +0.5 as well, and the old spelling cannot say so.
PIN_LEVELS: dict[str, dict[str, tuple[tuple[str, Any], ...]]] = {
    "ppo": {
        "readout_width": (("wide", "per_neuron"),),
        "forward_pass_depth": (("d2", 2), ("d3", 3), ("d6", 6)),
        "initial_log_std": (("lsm10", -1.0), ("lsm05", -0.5), ("lsp05", 0.5)),
    },
    "reading": {
        "readout_width": (("wide", "per_neuron"),),
        "forward_pass_depth": (("d2", 2), ("d3", 3), ("d6", 6)),
        "initial_log_std": (("lsm15", -1.5), ("lsm05", -0.5), ("ls00", 0.0)),
        "plasticity_rate": (("r1e4", 0.0001), ("r1e2", 0.01)),
        "trace_decay": (("td05", 0.5), ("td099", 0.99)),
    },
}
HALVES = tuple(PIN_LEVELS)

# ── Stems ────────────────────────────────────────────────────────────────────────────────────
_PPO = "connectomeppo_small_continuous2d_fick_adaptive_klinotaxis_hard350"
_READ = f"{_PPO}_eprop_readout_only"
_READ_FROZEN = f"{_PPO}_eprop_frozen"

CENTRE_STEMS: dict[str, dict[str, str]] = {
    "ppo": {
        "wt_learn": _PPO,
        "rn_learn": f"{_PPO}_rewired_null",
        "wt_frozen": f"{_PPO}_frozen",
        "rn_frozen": f"{_PPO}_rewired_null_frozen",
    },
    "reading": {
        "wt_learn": _READ,
        "rn_learn": f"{_READ}_rewired_null",
        "wt_frozen": _READ_FROZEN,
        "rn_frozen": f"{_READ_FROZEN}_rewired_null",
    },
}

# Committed configs whose spelling predates this panel. The e-prop family puts the variant BEFORE
# `_rewired_null`; A.1's convention appends it last. Both are in the tree, which is why the stem map
# is an explicit table rather than a rule: a regex over suffixes is the trap here, not the shortcut.
_COMMITTED_STEMS: dict[tuple[str, str, str], str] = {
    ("reading", "rn_learn", "wide"): f"{_READ}_wide_rewired_null",
    ("reading", "rn_frozen", "wide"): f"{_READ_FROZEN}_wide_rewired_null",
    ("reading", "rn_learn", "r1e4"): f"{_READ}_r1e4_rewired_null",
    ("reading", "rn_learn", "r1e2"): f"{_READ}_r1e2_rewired_null",
}


def _levels(half: str) -> tuple[tuple[str, str, Any], ...]:
    """Every off-centre level of ``half`` as ``(pin, suffix, value)``, in declaration order."""
    return tuple(
        (pin, suffix, value) for pin, levels in PIN_LEVELS[half].items() for suffix, value in levels
    )


def arms_at(half: str, suffix: str) -> tuple[str, ...]:
    """Which arms a level carries: four at a construction level, two at a learning-only one."""
    if suffix == CENTRE:
        return ARMS
    pin = next(p for p, s, _ in _levels(half) if s == suffix)
    return ARMS if pin in CONSTRUCTION_PINS else LEARNING_ARMS


def stem_for(half: str, arm: str, suffix: str) -> str:
    """Build the config stem for one arm at one level."""
    if suffix == CENTRE:
        return CENTRE_STEMS[half][arm]
    committed = _COMMITTED_STEMS.get((half, arm, suffix))
    if committed is not None:
        return committed
    return f"{CENTRE_STEMS[half][arm]}_{suffix}"


def _arms_by_stem() -> dict[str, tuple[str, str, str]]:
    """Config stem -> ``(half, arm, level suffix)``, built by a loop and never by a regex."""
    out: dict[str, tuple[str, str, str]] = {}
    for half in HALVES:
        for suffix in (CENTRE, *(s for _, s, _ in _levels(half))):
            for arm in arms_at(half, suffix):
                stem = stem_for(half, arm, suffix)
                if stem in out:
                    msg = f"stem {stem!r} is claimed by {out[stem]} and by {(half, arm, suffix)}"
                    raise ValueError(msg)
                out[stem] = (half, arm, suffix)
    return out


ARM_BY_STEM: dict[str, tuple[str, str, str]] = _arms_by_stem()

# ── Metric ───────────────────────────────────────────────────────────────────────────────────
# `episodes_to_30pct_success` is right-censored at the horizon and the interaction is a difference
# of differences, which the metric rule forbids unless censoring is comparable across the cells it
# spans. So the RULE is fixed here and the CHOICE follows the data: censoring counted per level,
# never pooled. Both metrics are reported either way.
CENSORED_METRIC = "episodes_to_30pct_success"
UNCENSORED_METRIC = "auc_success"
CENSORING_TOLERANCE = 0.10


class PanelError(ValueError):
    """The panel on disk is not the panel this module scores."""


def build_manifest(campaign_dir: Path, path: Path, half: str, seeds: tuple[int, ...]) -> Path:
    """Write ``<half> <arm> <suffix> <seed> <out>`` for every run of ``half`` in the campaign."""
    logs = campaign_dir / "logs"
    if not logs.is_dir():
        logs = campaign_dir
    seen: set[tuple[str, str, int]] = set()
    lines: list[str] = []
    for log in sorted(logs.glob("*.log")):
        stem, sep, seed_part = log.stem.rpartition("-seed")
        if not sep:
            msg = f"{log.name} has no -seedN suffix, so it cannot be placed in the panel"
            raise PanelError(msg)
        entry = ARM_BY_STEM.get(stem)
        if entry is None:
            msg = f"{log.name} names config {stem!r}, which this panel does not have"
            raise PanelError(msg)
        log_half, arm, suffix = entry
        if log_half != half:
            continue
        seed = int(seed_part)
        if seed not in seeds:
            continue
        key = (arm, suffix, seed)
        if key in seen:
            msg = f"{key} appears twice in {campaign_dir}"
            raise PanelError(msg)
        seen.add(key)
        resolved = log.resolve()
        try:
            out = str(resolved.relative_to(wp.REPO))
        except ValueError:
            out = str(resolved)
        lines.append(f"{arm} {suffix} {seed} {out}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(
    manifest: Path,
    half: str,
    seeds: tuple[int, ...],
    only_levels: tuple[str, ...] | None = None,
) -> None:
    """Refuse a partial panel.

    The instruments only WARN on a gap, and a surface whose consequence is which pins earn a full
    crossing must not be read off whatever finished. The gate is per level, and it knows that a
    learning-only level carries two arms rather than four: demanding four everywhere would refuse a
    correct panel as loudly as it refuses a broken one.

    ``only_levels`` names what the campaign was meant to cover, so a pilot -- a subset of levels by
    design -- is not refused for being what it is. It defaults to the whole half.
    """
    have: dict[tuple[str, str], set[int]] = {}
    for raw in manifest.read_text().splitlines():
        arm, suffix, seed, _ = raw.split()
        have.setdefault((arm, suffix), set()).add(int(seed))
    missing: list[str] = []
    wanted = only_levels if only_levels is not None else (CENTRE, *(s for _, s, _ in _levels(half)))
    for suffix in wanted:
        for arm in arms_at(half, suffix):
            gap = set(seeds) - have.get((arm, suffix), set())
            if gap:
                missing.append(f"{suffix}/{arm}: {sorted(gap)}")
    if missing:
        msg = f"{half} panel is incomplete — " + "; ".join(missing)
        raise PanelError(msg)


# ── Scoring ──────────────────────────────────────────────────────────────────────────────────
def _level_lines(manifest: Path, suffix: str) -> list[tuple[str, int, str]]:
    """``(arm, seed, out)`` for one level, read off this module's own manifest."""
    rows: list[tuple[str, int, str]] = []
    for raw in manifest.read_text().splitlines():
        arm, row_suffix, seed, out = raw.split()
        if row_suffix == suffix:
            rows.append((arm, int(seed), out))
    return rows


def score_level(manifest: Path, half: str, suffix: str, tmp_dir: Path) -> dict[str, Any]:
    """Score one level's wild-vs-rewired contrast through the instrument built for that half.

    The PPO half goes through ``wiring_premise``, whose ``EFFICIENCY_ARMS`` is keyed on the block-V
    arm names; the reading half calls the efficiency module directly with its own labels. Both
    modules are used unmodified, and neither learner's arms are relabelled as the other's.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = _level_lines(manifest, suffix)
    if half == "ppo":
        cell_manifest = tmp_dir / f"_wp_{suffix}.txt"
        cell_manifest.write_text(
            "\n".join(f"{CELL} {_WP_ARM[arm]} {seed} {out}" for arm, seed, out in rows) + "\n",
        )
        report = wp.efficiency_contrast(cell_manifest, CELL, tmp_dir)
    else:
        eff_manifest = tmp_dir / f"_eff_{suffix}.txt"
        eff_manifest.write_text(
            "\n".join(f"{_EFF_ARM[arm]} {seed} {out}" for arm, seed, out in rows if arm in _EFF_ARM)
            + "\n",
        )
        report = wp.apply_min_effect(eff.analyse(eff_manifest))
    if report is None:
        msg = f"{half} level {suffix!r} produced no paired contrast"
        raise PanelError(msg)
    return report


def _per_seed_gap(report: dict[str, Any], metric: str) -> dict[int, float]:
    """Wild minus rewired, per seed, on one metric -- read off the instrument's own per-seed block.

    Orientation is the instrument's: positive means the wild type is better, whichever direction the
    raw metric improves in.
    """
    per_seed = report["per_seed"]
    wild, rewired = per_seed[eff._WILD], per_seed[eff._REWIRED]
    higher_is_better = eff._METRICS[metric]
    out: dict[int, float] = {}
    for seed in sorted(set(wild) & set(rewired)):
        delta = float(wild[seed][metric]) - float(rewired[seed][metric])
        out[int(seed)] = delta if higher_is_better else -delta
    return out


def wiring_gap(report: dict[str, Any], metric: str) -> dict[str, Any]:
    """Measure the wiring effect at one level -- the quantity whose SIGN triggers a crossing.

    Reported beside the interaction and never in place of it: this is a reading at a point, and the
    claim that a pin MOVED the effect is the interaction's to make.
    """
    gap = _per_seed_gap(report, metric)
    seeds = sorted(gap)
    return {
        "metric": metric,
        "n_seeds": len(seeds),
        "gap_mean": sum(gap[s] for s in seeds) / len(seeds) if seeds else None,
        "per_seed": gap,
        "test": wp.paired_seed_wilcoxon_bootstrap([gap[s] for s in seeds]),
    }


def interaction(centre: dict[str, Any], level: dict[str, Any], metric: str) -> dict[str, Any]:
    """How much the wiring effect moves at this level -- the primary.

    Paired by seed, so each pair is one seed's wiring gap at the level minus the same seed's gap at
    the centre. A pin that moves both wirings equally cancels here, which is the point: that is a
    fact about the pin and not about the wiring.
    """
    base_gap = _per_seed_gap(centre, metric)
    level_gap = _per_seed_gap(level, metric)
    seeds = sorted(set(base_gap) & set(level_gap))
    deltas = {s: level_gap[s] - base_gap[s] for s in seeds}
    return {
        "metric": metric,
        "n_pairs": len(seeds),
        "centre_gap_mean": sum(base_gap[s] for s in seeds) / len(seeds) if seeds else None,
        "level_gap_mean": sum(level_gap[s] for s in seeds) / len(seeds) if seeds else None,
        "interaction_mean": sum(deltas.values()) / len(deltas) if deltas else None,
        "per_seed": deltas,
        "test": wp.paired_seed_wilcoxon_bootstrap([deltas[s] for s in seeds]),
    }


def learning_gates(
    manifest: Path,
    half: str,
    seeds: tuple[int, ...],
    suffix: str,
) -> dict[str, Any]:
    """Record each arm's plateau against its own frozen floor, and whether the level saturates.

    Two separate obligations live here and neither should rest on a reader's arithmetic.

    The **gate**: a wiring contrast is unreadable unless the claim-carrying arm beats its own floor
    on the same seeds. Recorded per arm rather than asserted, so a level that fails is reported as a
    gate failure -- itself a sensitivity result about that setting -- instead of being read as a
    wiring finding.

    **Saturation**: where both learning arms sit above the instrument's own ceiling, time-to-
    competence compresses and an abolished gap means "two arms tied at the top" rather than "the
    wiring stopped mattering". The instrument applies this on its peak axis; the efficiency axis
    decides this cell, so the flag has to be carried explicitly or the distinction is lost.
    """
    import t7_continuous_ranking as t7  # pyright: ignore[reportMissingImports]

    rows: dict[tuple[str, str], dict[int, Path]] = {}
    for raw in manifest.read_text().splitlines():
        arm, row_suffix, seed, log_path = raw.split()
        rows.setdefault((arm, row_suffix), {})[int(seed)] = wp.REPO / log_path

    floor_level = suffix if len(arms_at(half, suffix)) == 4 else CENTRE
    out: dict[str, Any] = {"floor_level": floor_level}
    plateaus: list[float] = []
    for wiring in ("wt", "rn"):
        learn: list[float] = []
        floor: list[float] = []
        deltas: list[float] = []
        beats = 0
        for seed in seeds:
            lp = rows.get((f"{wiring}_learn", suffix), {}).get(seed)
            fp = rows.get((f"{wiring}_frozen", floor_level), {}).get(seed)
            lv = t7.plateau_tail(lp) if lp else None
            fv = t7.plateau_tail(fp) if fp else None
            if lv is None or fv is None:
                continue
            learn.append(lv[0])
            floor.append(fv[0])
            deltas.append(lv[0] - fv[0])
            beats += int(lv[0] > fv[0])
        mean_learn = sum(learn) / len(learn) if learn else float("nan")
        plateaus.append(mean_learn)
        # The gate is the instrument's own paired test of learning against its floor, not a
        # count of seeds. An all-seeds rule is stricter than anything registered, and it would
        # report an arm that plainly learns as a gate failure on one unlucky seed.
        test = wp.paired_seed_wilcoxon_bootstrap(deltas) if deltas else None
        out[wiring] = {
            "plateau_success": mean_learn,
            "floor_success": sum(floor) / len(floor) if floor else float("nan"),
            "beats_floor_seeds": beats,
            "n_seeds": len(learn),
            "vs_floor": test,
        }
    out["gate_passes"] = all(
        (out[w]["vs_floor"] or {}).get("ci_lo", 0.0) > 0.0 for w in ("wt", "rn")
    )
    out["saturated"] = bool(plateaus) and all(v >= wp.SATURATION_SUCCESS for v in plateaus)
    out["saturation_bar"] = wp.SATURATION_SUCCESS
    return out


def two_sided(p: float) -> float:
    """Fold the instrument's one-sided p into a two-sided one.

    ``paired_seed_wilcoxon_bootstrap`` tests ``a > b``, so a p near 1.0 means a strongly NEGATIVE
    delta rather than a null one. The registered minimum is stated in both directions -- a level
    that amplifies the wiring effect is the same finding with the opposite sign -- so the family
    correction has to see a statistic that treats the two tails alike.
    """
    return min(1.0, 2.0 * min(p, 1.0 - p))


def apply_family_correction(levels: dict[str, Any]) -> dict[str, Any]:
    """BH-FDR the interaction tests across the pin family, every test computed before any is read.

    The family is one half's levels. Correcting within it is what the registration commits to, and
    doing it after the branches were assigned would let the reading choose its own family.
    """
    entries: list[tuple[str, str]] = []
    pvals: list[float] = []
    for suffix, entry in levels.items():
        for metric in [
            k for k in entry if isinstance(entry[k], dict) and "interaction" in entry[k]
        ]:
            entries.append((suffix, metric))
            pvals.append(two_sided(entry[metric]["interaction"]["test"]["wilcoxon_p"]))
    for (suffix, metric), q, raw in zip(entries, wp.bh_fdr(pvals), pvals, strict=True):
        test = levels[suffix][metric]["interaction"]["test"]
        test["two_sided_p"] = raw
        test["bh_q"] = q
    return levels


def censoring_rates(report: dict[str, Any]) -> dict[str, float]:
    """Per-arm crossing rate, through the instrument's own function rather than reimplemented."""
    return {arm: wp.crossing_rate(report, arm) for arm in (eff._WILD, eff._REWIRED)}


def choose_metric(rates_by_level: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the registered censoring rule. Fixed before any rate is known; see the module docstring.

    The censored metric may carry a difference of differences only when censoring is comparable
    across every cell of the design. Where it is not, the uncensored metric carries it and the
    censored one is reported beside it -- which is what the metric requirement asks of a departure.
    """
    observed = [r for rates in rates_by_level.values() for r in rates.values()]
    spread = max(observed) - min(observed) if observed else 0.0
    equal = spread <= CENSORING_TOLERANCE
    return {
        "primary_metric": CENSORED_METRIC if equal else UNCENSORED_METRIC,
        "reported_beside": UNCENSORED_METRIC if equal else CENSORED_METRIC,
        "crossing_rates_by_level": rates_by_level,
        "crossing_rate_spread": spread,
        "tolerance": CENSORING_TOLERANCE,
        "censoring_comparable": equal,
        "why": (
            "censoring is comparable across the surface's levels, so the registered censored "
            "metric carries the interaction"
            if equal
            else (
                "censoring differs across the surface's levels by more than the registered "
                "tolerance, so a difference of differences on the censored metric is not "
                "interpretable and the uncensored metric carries it"
            )
        ),
    }


# ── Frozen-substrate drift (the reading half's obligation) ───────────────────────────────────
# The reading learner leaves ``w_chem`` fixed, so the requirement governing a contrast under a
# learner that does not write the wiring applies in full: the fixed tensors are compared against a
# control in which nothing learned, ON EVERY SCORED SEED, and any non-zero drift -- or evidence
# missing for any seed -- returns VOID rather than "it held".
#
# Evidence is ``<exports_path>/weights/final.pt``, which the runner auto-saves unconditionally; what
# it needs is ``--track-experiment``, without which no export path is recorded and this reads nothing
# for every run. The comparator at a learning-only level is the CENTRE's frozen arm, which is correct
# rather than convenient: ``w_chem`` at a given seed is the same draw whatever the rate or the decay
# says, and the learning arm never writes it.
_DRIFT_TOLERANCE = 1e-6


def substrate_drift(
    manifest: Path,
    half: str,
    seeds: tuple[int, ...],
    only_levels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compare each learning arm's chemical matrix against its frozen floor, seed by seed."""
    import l4_reduced_perturbation as rp  # pyright: ignore[reportMissingImports]

    rows: dict[tuple[str, str], dict[int, Path]] = {}
    for raw in manifest.read_text().splitlines():
        arm, suffix, seed, out = raw.split()
        rows.setdefault((arm, suffix), {})[int(seed)] = wp.REPO / out

    wanted = only_levels if only_levels is not None else (CENTRE, *(s for _, s, _ in _levels(half)))
    per_level: dict[str, Any] = {}
    for suffix in wanted:
        # A construction level has its own floor; a learning-only level uses the centre's, which
        # is the same draw at that seed because neither rule pin can reach a frozen arm.
        floor_level = suffix if len(arms_at(half, suffix)) == 4 else CENTRE
        entry: dict[str, Any] = {"floor_level": floor_level}
        for wiring in ("wt", "rn"):
            values: list[float] = []
            unread: list[int] = []
            for seed in seeds:
                learning = rows.get((f"{wiring}_learn", suffix), {}).get(seed)
                frozen = rows.get((f"{wiring}_frozen", floor_level), {}).get(seed)
                a = rp._chemical_weights(learning) if learning else None
                b = rp._chemical_weights(frozen) if frozen else None
                if a is None or b is None or a.shape != b.shape:
                    unread.append(seed)
                    continue
                base = b.float().norm()
                values.append(float((a - b).float().norm() / (base or 1.0)))
            entry[wiring] = {
                "mean_relative": sum(values) / len(values) if values else float("nan"),
                "max_relative": max(values) if values else float("nan"),
                "n_read": len(values),
                # Complete evidence, not merely available evidence: with some checkpoints missing,
                # a mean over whatever happened to be on disk would report the substrate frozen on
                # the strength of the runs that survived.
                "seeds_unread": unread,
            }
        complete = all(not entry[w]["seeds_unread"] for w in ("wt", "rn"))
        frozen_ok = complete and all(
            entry[w]["max_relative"] < _DRIFT_TOLERANCE for w in ("wt", "rn")
        )
        entry["evidence_complete"] = complete
        entry["substrate_frozen"] = bool(frozen_ok)
        per_level[suffix] = entry

    return {
        "tolerance": _DRIFT_TOLERANCE,
        "levels": per_level,
        "void": not all(e["substrate_frozen"] for e in per_level.values()),
    }


def score(
    campaign_dir: Path,
    half: str,
    out_dir: Path,
    seeds: tuple[int, ...] | None = None,
    only_levels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Read one half's surface: every level's wiring gap, and its interaction with the centre."""
    seeds = seeds if seeds is not None else SEEDS_BY_HALF[half]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(campaign_dir, out_dir / f"manifest-{half}.txt", half, seeds)
    require_complete(manifest, half, seeds, only_levels)

    suffixes = (
        list(only_levels)
        if only_levels is not None
        else [CENTRE, *(s for _, s, _ in _levels(half))]
    )
    reports = {
        suffix: score_level(manifest, half, suffix, out_dir / f"tmp-{half}-{suffix}")
        for suffix in suffixes
    }
    rates = {suffix: censoring_rates(report) for suffix, report in reports.items()}

    levels: dict[str, Any] = {}
    for pin, suffix, value in _levels(half):
        if suffix not in suffixes:
            continue
        # The metric choice is PER LEVEL, because the cells an interaction spans are the centre and
        # that level -- not the whole surface. Pooling the comparison would let one level whose arms
        # censor differently void the censored metric everywhere, including at levels where it is
        # perfectly interpretable, and the requirement's scope is the contrast rather than the
        # campaign.
        choice = choose_metric({CENTRE: rates[CENTRE], suffix: rates[suffix]})
        entry: dict[str, Any] = {
            "pin": pin,
            "value": value,
            "carries_own_floor": pin in CONSTRUCTION_PINS,
            "metric_choice": choice,
            "gates": learning_gates(manifest, half, seeds, suffix),
        }
        for metric in (choice["primary_metric"], choice["reported_beside"]):
            entry[metric] = {
                "interaction": interaction(reports[CENTRE], reports[suffix], metric),
                "wiring_gap": wiring_gap(reports[suffix], metric),
            }
        levels[suffix] = entry

    apply_family_correction(levels)

    # Kept beside the per-level choices as a summary of the whole surface's censoring, so a reader
    # can see at a glance whether any level drove the censored metric out.
    metric_choice = choose_metric(rates)

    centre: dict[str, Any] = {"gates": learning_gates(manifest, half, seeds, CENTRE)}
    for metric in (CENSORED_METRIC, UNCENSORED_METRIC):
        centre[metric] = {"wiring_gap": wiring_gap(reports[CENTRE], metric)}

    # The reading learner leaves `w_chem` fixed, so its surface is unreadable without drift
    # evidence on every scored seed. Run here rather than left to a separate step: an obligation
    # whose failure mode is VOID should not be possible to forget. The PPO half owes none of it --
    # PPO writes the chemical matrix by design, and the check reads ~1.0 there, which is what
    # establishes that a 0.00 on the reading half means something.
    drift = substrate_drift(manifest, half, seeds, only_levels) if half == "reading" else None

    return {
        "half": half,
        "cell": CELL,
        "seeds": list(seeds),
        "substrate_drift": drift,
        "metric_choice": metric_choice,
        "centre": centre,
        "levels": levels,
        "verdicts": {suffix: report.get("verdict") for suffix, report in reports.items()},
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per (level, metric, seed): the per-seed unit both tests consume."""
    # The primary is per level now, so the CSV says which metric carried each row's contrast.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        # csv defaults to CRLF, which would make every regeneration read as a whole-file diff.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "half",
                "level",
                "pin",
                "metric",
                "is_primary_metric",
                "seed",
                "wiring_gap",
                "interaction",
            ],
        )
        for suffix, entry in result["levels"].items():
            for metric in sorted(
                k for k in entry if isinstance(entry[k], dict) and "interaction" in entry[k]
            ):
                gaps = entry[metric]["wiring_gap"]["per_seed"]
                inter = entry[metric]["interaction"]["per_seed"]
                for seed in sorted(gaps, key=int):
                    writer.writerow(
                        [
                            result["half"],
                            suffix,
                            entry["pin"],
                            metric,
                            metric == entry["metric_choice"]["primary_metric"],
                            seed,
                            f"{gaps[seed]:.6f}",
                            f"{inter.get(seed, float('nan')):.6f}",
                        ],
                    )
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: score one half of the surface from its campaign directory."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--campaign", type=Path, required=True, help="campaign directory holding logs/")
    ap.add_argument("--half", choices=HALVES, required=True, help="which half to score")
    ap.add_argument("--out-dir", type=Path, required=True, help="scratch directory for manifests")
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="score the pilot band instead of the panel",
    )
    ap.add_argument("--out", type=Path, help="write the analysis JSON here instead of stdout")
    ap.add_argument("--csv", type=Path, help="write the per-seed CSV here")
    args = ap.parse_args(argv)

    seeds = PILOT_SEEDS if args.pilot else SEEDS_BY_HALF[args.half]
    only_levels = PILOT_LEVELS[args.half] if args.pilot else None
    result = score(args.campaign, args.half, args.out_dir, seeds, only_levels)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    else:
        print(payload)
    if args.csv:
        write_csv(result, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
